import sqlite3
import os
import openai
import datetime
import time

import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_RETRIES = int(os.getenv("LLM_RETRIES", 3))  # type: ignore
LLM_RETRY_DELAY_SECONDS = int(os.getenv("LLM_RETRY_DELAY_SECONDS", 2))  # type: ignore


class LLMManager:
    def __init__(self, db_filename, api_key):
        self.db_filename = db_filename
        self.api_key = api_key
        openai.api_key = self.api_key
        self.retries = LLM_RETRIES
        self.retry_delay_seconds = LLM_RETRY_DELAY_SECONDS
        self.connection = sqlite3.connect(self.db_filename)
        self.cursor = self.connection.cursor()
        self.create_tables()
        self.validate_tables()

    def create_tables(self):
        # Create prompts table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                system_message TEXT,
                model TEXT
            )
        """
        )

        # Create responses table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER,
                response TEXT,
                error TEXT,
                status TEXT,
                refusal TEXT,
                finish_reason TEXT,
                timestamp DATETIME,
                prompt_tokens INTEGER,
                response_tokens INTEGER,
                FOREIGN KEY(prompt_id) REFERENCES prompts(id)
            )
        """
        )

        self.connection.commit()

    def validate_tables(self):
        # Validate prompts table columns
        self.cursor.execute("PRAGMA table_info(prompts)")
        prompts_columns = self.cursor.fetchall()
        expected_prompts_columns = [
            ("id", "INTEGER"),
            ("prompt", "TEXT"),
            ("system_message", "TEXT"),
            ("model", "TEXT"),
        ]
        for expected_column in expected_prompts_columns:
            found = False
            for column in prompts_columns:
                if (
                    column[1] == expected_column[0]
                    and column[2].upper() == expected_column[1]
                ):
                    found = True
                    break
            if not found:
                raise Exception(
                    f"Prompts table schema does not match expected schema. Missing column: {expected_column[0]} {expected_column[1]}"
                )

        # Validate responses table columns
        self.cursor.execute("PRAGMA table_info(responses)")
        responses_columns = self.cursor.fetchall()
        expected_responses_columns = [
            ("id", "INTEGER"),
            ("prompt_id", "INTEGER"),
            ("response", "TEXT"),
            ("error", "TEXT"),
            ("status", "TEXT"),
            ("refusal", "TEXT"),
            ("finish_reason", "TEXT"),
            ("timestamp", "DATETIME"),
            ("prompt_tokens", "INTEGER"),
            ("response_tokens", "INTEGER"),
        ]
        for expected_column in expected_responses_columns:
            found = False
            for column in responses_columns:
                if (
                    column[1] == expected_column[0]
                    and column[2].upper() == expected_column[1]
                ):
                    found = True
                    break
            if not found:
                raise Exception(
                    f"Responses table schema does not match expected schema. Missing column: {expected_column[0]} {expected_column[1]}"
                )

    def generate_response(
        self,
        prompt,
        model="gpt-4o",
        system_message="You are a helpful assistant.",
        store=True,
    ):
        prompt_id = None
        error_message = None
        response_json_text = None
        status = "failure"
        refusal = None
        finish_reason = None
        content = None
        prompt_tokens = None
        response_tokens = None
        try:
            if store:
                # Insert prompt into prompts table
                self.cursor.execute(
                    """
                    INSERT INTO prompts (prompt, system_message, model) VALUES (?, ?, ?)
                """,
                    (prompt, system_message, model),
                )
                self.connection.commit()
                prompt_id = self.cursor.lastrowid

            # Call OpenAI API
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            attempt = 0

            while attempt < self.retries:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    response_json_text = response.model_dump_json()
                    content = response.choices[0].message.content
                    refusal = response.choices[0].message.refusal
                    finish_reason = response.choices[0].finish_reason
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= self.retries:
                        # If maximum attempts reached, re-raise the exception or handle it
                        raise RuntimeError(
                            f"Failed to generate response summary after {self.retries} attempts."
                        ) from e
                    else:
                        # Optional: Log the error or provide feedback
                        logger.warning(f"Attempt {attempt} failed: {e}. Retrying...")
                        time.sleep(self.retry_delay_seconds)

            status = "success"
        except Exception as e:
            error_message = str(e)
            status = "failure"

        if store and prompt_id is not None:
            timestamp = datetime.datetime.now()
            self.cursor.execute(
                """
                INSERT INTO responses (
                    prompt_id, response, error, status, refusal, finish_reason,
                    timestamp, prompt_tokens, response_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    prompt_id,
                    response_json_text,
                    error_message,
                    status,
                    refusal,
                    finish_reason,
                    timestamp,
                    prompt_tokens,
                    response_tokens,
                ),
            )
            self.connection.commit()

        return {
            "content": content,
            "status": status,
            "finish_reason": finish_reason,
            "refusal": refusal,
        }

    def close(self):
        self.connection.close()
