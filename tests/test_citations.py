"""Unit tests for citation normalization and extraction."""

import unittest
from src.utils.citations import normalize_citations, extract_cited_sources


class TestCitations(unittest.TestCase):

    def test_normalize_raw_chunk_id(self):
        text = "A Beacon evaluates every 5 minutes [spec-beacons:1]."
        expected = "A Beacon evaluates every 5 minutes [spec-beacons:1: Beacons: Alerting Specification]."
        self.assertEqual(normalize_citations(text), expected)

    def test_normalize_chinese_brackets(self):
        text = "Funnels allow up to 12 steps 【spec-funnels-cohorts:0】."
        expected = "Funnels allow up to 12 steps [spec-funnels-cohorts:0: Funnels and Cohorts Specification]."
        self.assertEqual(normalize_citations(text), expected)

    def test_preserve_existing_valid_citation(self):
        text = "Trails is not available on Starter [pricing-plans:0: Plans, Pricing and Limits]."
        expected = "Trails is not available on Starter [pricing-plans:0: Plans, Pricing and Limits]."
        self.assertEqual(normalize_citations(text), expected)

    def test_normalize_with_retrieved_chunks(self):
        chunks = [
            {
                "metadata": {
                    "chunk_id": "custom-doc:0",
                    "title": "Custom Document Title",
                },
                "text": "sample text",
            }
        ]
        text = "Custom feature explained [custom-doc:0]."
        expected = "Custom feature explained [custom-doc:0: Custom Document Title]."
        self.assertEqual(normalize_citations(text, chunks), expected)

    def test_extract_cited_sources(self):
        text = (
            "Beacons alert users [spec-beacons:0: Beacons: Alerting Specification]. "
            "Ingest batch size is 500 [spec-ingest-api:1]."
        )
        sources = extract_cited_sources(text)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["chunk_id"], "spec-beacons:0")
        self.assertEqual(sources[0]["title"], "Beacons: Alerting Specification")
        self.assertEqual(sources[1]["chunk_id"], "spec-ingest-api:1")
        self.assertEqual(sources[1]["title"], "Event Ingestion API Specification")

    def test_no_citations(self):
        text = "Hello, how can I help you today?"
        self.assertEqual(normalize_citations(text), text)
        self.assertEqual(extract_cited_sources(text), [])


if __name__ == "__main__":
    unittest.main()
