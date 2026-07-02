from pathlib import Path
from unittest.mock import patch, MagicMock
from src.member1_ingestion.parser import DoclingLayoutParser
from src.member1_ingestion.schemas import CandidateProfile


@patch("src.member1_ingestion.parser.DocumentConverter")
def test_docling_parser_stub(mock_converter_cls, tmp_path):
    # Setup mock document and converter to prevent actual deep learning model loading/downloads
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = "Unstructured resume content"
    mock_doc.iterate_items.return_value = []

    mock_result = MagicMock()
    mock_result.document = mock_doc

    mock_converter_inst = MagicMock()
    mock_converter_inst.convert.return_value = mock_result
    mock_converter_cls.return_value = mock_converter_inst

    # Setup temporary file as text so Docling parses it correctly without PDF binary expectations
    test_file = tmp_path / "resume_jane.txt"
    test_file.write_text("Unstructured resume content")

    parser = DoclingLayoutParser()
    result = parser.parse_document(test_file)

    assert "metadata" in result
    assert result["metadata"]["file_name"] == "resume_jane.txt"
    assert "raw_text" in result
    assert result["raw_text"] == "Unstructured resume content"


def test_candidate_profile_validation():
    profile = CandidateProfile(
        name="Jane Doe",
        skills=["Python", "SQL"],
        experience=[
            {"company": "Google", "role": "Software Engineer", "duration_months": 24}
        ],
        education=[],
    )
    assert profile.name == "Jane Doe"
    assert len(profile.experience) == 1
    assert profile.experience[0].company == "Google"


@patch("src.member1_ingestion.parser.DocumentConverter")
def test_docling_parser_fallback_text(mock_converter_cls, tmp_path):
    test_file = tmp_path / "fallback.txt"
    test_file.write_text("This is fallback text content")

    parser = DoclingLayoutParser()
    with patch.object(
        parser.converter, "convert", side_effect=Exception("Docling error")
    ):
        result = parser.parse_document(test_file)
        assert result["raw_text"] == "This is fallback text content"


def test_celery_task_retry():
    from src.member1_ingestion.tasks import ingest_resume_pipeline

    with patch("src.member1_ingestion.tasks.DoclingLayoutParser") as mock_parser_cls:
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_document.side_effect = Exception("Parsing failed")

        ingest_resume_pipeline.retry = MagicMock()
        try:
            ingest_resume_pipeline("dummy_path.pdf")
        except Exception:
            pass

        assert ingest_resume_pipeline.retry.called
