from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from meetscribe.summarization.provider import SummarizationProvider


class TestSummarizationProvider:
    def test_init_sets_endpoint(self):
        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        assert provider.model == "llama3"
        assert provider.base_url == "http://localhost:11434/v1"

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_list_models(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        model1 = MagicMock()
        model1.id = "llama3"
        model2 = MagicMock()
        model2.id = "mistral"
        mock_client.models.list.return_value = MagicMock(data=[model1, model2])

        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        models = provider.list_models()
        assert models == ["llama3", "mistral"]

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_summarize(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary here."))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        result = provider.summarize("You are a summarizer.", "Summarize this meeting.")
        assert result == "Summary here."
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a summarizer."},
                {"role": "user", "content": "Summarize this meeting."},
            ],
        )

    @patch("meetscribe.summarization.provider.OpenAI")
    def test_list_models_returns_empty_on_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.side_effect = Exception("Connection refused")

        provider = SummarizationProvider(
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        models = provider.list_models()
        assert models == []
