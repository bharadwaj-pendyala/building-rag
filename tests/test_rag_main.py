import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import rag.main as main


class TestRagMain(unittest.TestCase):
    @patch("rag.main.load_youtube_transcript", return_value=[])
    @patch("rag.main.load_csv", return_value=[])
    @patch("rag.main.load_pdf", return_value=[])
    @patch("rag.main.load_json", return_value=[])
    @patch("rag.main.load_txt", return_value=[])
    @patch("rag.main.split_text", return_value=[])
    @patch("rag.main.Chroma")
    @patch("rag.main.ChatOpenAI")
    @patch("rag.main.OpenAIEmbeddings")
    @patch("rag.main.RetrievalQA")
    def test_build_qa_chain_wires_components(
        self,
        mock_retrievalqa,
        mock_embeddings,
        mock_chat,
        mock_chroma,
        *_
    ):
        retriever = MagicMock()
        vector_store = MagicMock()
        vector_store.as_retriever.return_value = retriever
        mock_chroma.from_documents.return_value = vector_store

        expected_chain = object()
        mock_retrievalqa.from_chain_type.return_value = expected_chain

        chain = main.build_qa_chain()

        self.assertIs(chain, expected_chain)
        mock_embeddings.assert_called_once_with(api_key="test-key")
        mock_chat.assert_called_once_with(model="gpt-3.5-turbo")
        mock_retrievalqa.from_chain_type.assert_called_once()

    @patch("rag.main.build_qa_chain")
    @patch("builtins.input", side_effect=["What is RAG?", "exit"])
    @patch("builtins.print")
    def test_run_cli_handles_query_then_exit(self, mock_print, _mock_input, mock_build):
        qa = MagicMock()
        qa.return_value = {"result": "A retrieval-augmented generation approach."}
        mock_build.return_value = qa

        main.run_cli()

        qa.assert_called_once_with({"query": "What is RAG?"})
        mock_print.assert_any_call("Response:", "A retrieval-augmented generation approach.")


if __name__ == "__main__":
    unittest.main()
