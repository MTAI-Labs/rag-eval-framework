"""Retrieval metrics against synthetic traces with hand-computable answers."""

from conftest import chunk, item, ref, trace

from rag_eval.metrics import retrieval


def test_hit_recall_and_mrr_for_a_perfect_retrieval():
    gold = item("q1", ref("dr", "2026-06-22", 3))
    t = trace("q1", [chunk(1, "dr_2026-06-22", 3), chunk(2, "dn_2026-08-04", 9)])

    result = retrieval.score_question(gold, t, k_values=(1, 3, 5))
    assert result.hit_at_k == {1: True, 3: True, 5: True}
    assert result.recall_at_k == {1: True, 3: True, 5: True}
    assert result.first_correct_rank == 1
    assert result.reciprocal_rank == 1.0


def test_right_sitting_wrong_page_hits_but_does_not_recall():
    # The distinction the scorecard lives on: retrieval found the sitting but
    # not the evidence, which is a chunking problem, not a retriever problem.
    gold = item("q1", ref("dr", "2026-06-22", 3))
    t = trace("q1", [chunk(1, "dr_2026-06-22", 99)])

    result = retrieval.score_question(gold, t, k_values=(1,))
    assert result.hit_at_k[1] is True
    assert result.recall_at_k[1] is False


def test_hit_at_k_respects_the_k_window():
    gold = item("q1", ref("dr", "2026-06-22", 3))
    t = trace("q1", [
        chunk(1, "dn_2026-08-04", 1),
        chunk(2, "dn_2026-08-04", 2),
        chunk(3, "dr_2026-06-22", 3),
    ])

    result = retrieval.score_question(gold, t, k_values=(1, 3))
    assert result.hit_at_k == {1: False, 3: True}
    assert result.reciprocal_rank == 1 / 3


def test_chunks_missing_metadata_can_never_match():
    gold = item("q1", ref("dr", "2026-06-22", 3))
    t = trace("q1", [chunk(1, None, None), chunk(2, None, None)])

    result = retrieval.score_question(gold, t, k_values=(5,))
    assert result.hit_at_k[5] is False
    assert result.chunks_missing_metadata == 2


def test_page_citation_accuracy_accepts_one_page_of_a_gold_range():
    # The workbook records "ms. 15-16"; citing page 15 is correct behaviour.
    gold = item("q2", ref("dn", "2026-08-04", 15, 16))
    t = trace("q2", [chunk(1, "dn_2026-08-04", 15)],
              cited=[ref("dn", "2026-08-04", 15)])

    result = retrieval.score_question(gold, t, k_values=(5,))
    assert result.citation_sitting_match is True
    assert result.citation_page_match is True


def test_citing_the_right_sitting_but_the_wrong_page_fails_page_accuracy():
    gold = item("q2", ref("dn", "2026-08-04", 15, 16))
    t = trace("q2", [chunk(1, "dn_2026-08-04", 15)], cited=[ref("dn", "2026-08-04", 40)])

    result = retrieval.score_question(gold, t, k_values=(5,))
    assert result.citation_sitting_match is True
    assert result.citation_page_match is False


def test_unreferenced_questions_are_excluded_not_scored_as_misses():
    scored = [
        retrieval.score_question(
            item("q1", ref("dr", "2026-06-22", 3)),
            trace("q1", [chunk(1, "dr_2026-06-22", 3)], cited=[ref("dr", "2026-06-22", 3)]),
            k_values=(5,),
        ),
        retrieval.score_question(
            item("q4", None), trace("q4", [chunk(1, "dn_2026-08-04", 1)]), k_values=(5,)
        ),
    ]
    aggregate = retrieval.aggregate(scored, k_values=(5,))

    assert aggregate["questions"] == 2
    assert aggregate["scorable"] == 1
    assert aggregate["unscorable_no_reference"] == 1
    # 1/1, not 1/2: the unreferenced row must not drag the score down.
    assert aggregate["hit_rate@5"] == 1.0
    assert aggregate["page_citation_accuracy"] == 1.0


def test_mrr_is_the_mean_over_scorable_questions():
    scored = [
        retrieval.score_question(
            item("q1", ref("dr", "2026-06-22", 3)),
            trace("q1", [chunk(1, "dr_2026-06-22", 3)]), k_values=(5,)),
        retrieval.score_question(
            item("q2", ref("dn", "2026-08-04", 15)),
            trace("q2", [chunk(1, "dr_2026-06-22", 3), chunk(2, "dn_2026-08-04", 15)]),
            k_values=(5,)),
    ]
    assert retrieval.aggregate(scored, k_values=(5,))["mrr"] == 0.75  # (1 + 0.5) / 2


def test_metadata_health_flags_an_ingestion_problem():
    scored = [
        retrieval.score_question(
            item("q1", ref("dr", "2026-06-22", 3)),
            trace("q1", [chunk(1, None, None), chunk(2, "dr_2026-06-22", 3)]),
            k_values=(5,),
        )
    ]
    health = retrieval.aggregate(scored, k_values=(5,))["metadata_health"]
    assert health["retrieved_chunks"] == 2
    assert health["chunks_missing_sitting_id"] == 1
    assert health["chunks_with_sitting_id_pct"] == 50.0


def test_empty_aggregate_reports_none_not_zero():
    aggregate = retrieval.aggregate([], k_values=(5,))
    assert aggregate["hit_rate@5"] is None
    assert aggregate["mrr"] is None


# -- citation status: "didn't cite" is a finding, not missing data ---------
def test_citation_status_separates_the_four_outcomes():
    gold = ref("dr", "2026-06-22", 3)
    assert retrieval.citation_status([ref("dr", "2026-06-22", 3)], gold) == "page_match"
    assert retrieval.citation_status([ref("dr", "2026-06-22", 99)], gold) == "sitting_only"
    assert retrieval.citation_status([ref("dn", "2026-08-04", 3)], gold) == "wrong"
    assert retrieval.citation_status([], gold) == "none"


def test_no_citation_counts_as_a_miss_but_is_reported_separately():
    # Excluding uncited questions would flatter a RAG that simply stopped
    # citing; hiding them among the wrong-page failures would point whoever
    # reads the scorecard at the wrong fix.
    gold = item("q1", ref("dr", "2026-06-22", 3))
    scored = [
        retrieval.score_question(gold, trace("q1", [chunk(1, "dr_2026-06-22", 3)],
                                             cited=[ref("dr", "2026-06-22", 3)]), k_values=(5,)),
        retrieval.score_question(gold, trace("q2", [chunk(1, "dr_2026-06-22", 3)],
                                             cited=[]), k_values=(5,)),
    ]
    agg = retrieval.aggregate(scored, k_values=(5,))

    assert agg["page_citation_accuracy"] == 0.5      # the uncited one counts as a miss
    assert agg["no_citation_rate"] == 0.5            # and is visible on its own
    assert agg["citation_breakdown"] == {
        "page_match": 1, "sitting_only": 0, "wrong": 0, "none": 1,
    }


def test_the_breakdown_distinguishes_wrong_page_from_wrong_sitting():
    gold = item("q1", ref("dr", "2026-06-22", 3))
    scored = [
        retrieval.score_question(gold, trace("q1", [], cited=[ref("dr", "2026-06-22", 99)]),
                                 k_values=(5,)),
        retrieval.score_question(gold, trace("q2", [], cited=[ref("dn", "2026-08-04", 3)]),
                                 k_values=(5,)),
    ]
    agg = retrieval.aggregate(scored, k_values=(5,))
    assert agg["citation_breakdown"]["sitting_only"] == 1
    assert agg["citation_breakdown"]["wrong"] == 1
    assert agg["no_citation_rate"] == 0.0


def test_breakdown_totals_match_the_scorable_count():
    gold = item("q1", ref("dr", "2026-06-22", 3))
    scored = [retrieval.score_question(gold, trace(f"q{i}", [], cited=c), k_values=(5,))
              for i, c in enumerate([[ref("dr", "2026-06-22", 3)], [], [ref("dn", "2019-05-06", 1)]])]
    agg = retrieval.aggregate(scored, k_values=(5,))
    assert sum(agg["citation_breakdown"].values()) == agg["scorable"] == 3


# -- empty retrieval and errored traces ------------------------------------
def test_empty_retrieval_scores_zero_without_dividing_by_zero():
    # The RAG returned nothing at all. Every retrieval metric is a legitimate
    # miss, not missing data — the question was asked and answered badly.
    gold = item("q1", ref("dr", "2026-06-22", 3))
    r = retrieval.score_question(gold, trace("q1", [], cited=[]), k_values=(1, 5))

    assert r.scorable is True
    assert r.hit_at_k == {1: False, 5: False}
    assert r.recall_at_k == {1: False, 5: False}
    assert r.reciprocal_rank == 0.0
    assert r.first_correct_rank is None
    assert r.citation_status == "none"


def test_an_errored_trace_still_scores_as_a_miss():
    # An adapter failure is an ops problem, but the question still did not get
    # its evidence — retrieval must not quietly skip it.
    gold = item("q1", ref("dr", "2026-06-22", 3))
    r = retrieval.score_question(gold, trace("q1", [], error="HTTP 503"), k_values=(5,))
    agg = retrieval.aggregate([r], k_values=(5,))

    assert agg["scorable"] == 1
    assert agg["hit_rate@5"] == 0.0
    assert agg["no_citation_rate"] == 1.0


def test_a_question_with_no_gold_reference_is_excluded_from_citations_too():
    r = retrieval.score_question(item("q4", None), trace("q4", [], cited=[]), k_values=(5,))
    agg = retrieval.aggregate([r], k_values=(5,))

    assert agg["scorable"] == 0
    assert agg["unscorable_no_reference"] == 1
    assert agg["page_citation_accuracy"] is None      # no denominator, not zero
    assert agg["no_citation_rate"] is None
