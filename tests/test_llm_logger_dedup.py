from covxplore.llm_logger import LLMInteractionLogger


def interaction(source: str, index: int) -> dict:
    return {
        "call_index": index,
        "source": source,
        "model": "openai/model",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def test_cross_callback_duplicate_is_removed_but_real_retries_are_retained() -> None:
    logger = LLMInteractionLogger()
    logger._append_if_new(interaction("litellm", 1))
    logger._append_if_new(interaction("crewai_provider", 2))
    logger._append_if_new(interaction("litellm", 3))

    assert [item["call_index"] for item in logger.interactions] == [1, 3]


def test_logger_reads_crewai_direct_callback_usage_dict() -> None:
    logger = LLMInteractionLogger()

    logger.log_success_event(
        {"model": "deepseek-chat", "messages": []},
        {
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 23,
                "total_tokens": 124,
            }
        },
        0,
        0,
    )

    assert logger.interactions[0]["usage"] == {
        "prompt_tokens": 101,
        "completion_tokens": 23,
        "total_tokens": 124,
    }
