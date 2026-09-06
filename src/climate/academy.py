"""
Talaix Academy loader and logic.

Course and glossary content lives as config JSON registries
(``config/academy_course.json`` and ``config/academy_glossary.json``), loaded
in the same honest, fail-loud style as the other data registries.

No Flask imports. Quiz grading happens server-side; public course payloads have
correct answers and explanations stripped.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_COURSE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "academy_course.json"
)
_DEFAULT_GLOSSARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "academy_glossary.json"
)
_DEFAULT_KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "academy_knowledge.json"
)


class AcademyConfigError(RuntimeError):
    """Raised when course or glossary config cannot be loaded or is invalid."""


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _load_json(path: str, label: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise AcademyConfigError(f"{label} not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise AcademyConfigError(f"{label} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise AcademyConfigError(f"{label} unreadable: {exc}") from exc


def load_course(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the full academy course registry."""
    path = path or os.environ.get("HYDRASHIELD_ACADEMY_COURSE") or _DEFAULT_COURSE_PATH
    data = _load_json(path, "Academy course config")
    if not isinstance(data, dict) or "courses" not in data:
        raise AcademyConfigError("Academy course config must be an object with a 'courses' list")
    return data


def load_glossary(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the full academy glossary registry."""
    path = path or os.environ.get("HYDRASHIELD_ACADEMY_GLOSSARY") or _DEFAULT_GLOSSARY_PATH
    data = _load_json(path, "Academy glossary config")
    if not isinstance(data, dict) or "terms" not in data:
        raise AcademyConfigError("Academy glossary config must be an object with a 'terms' list")
    return data


def get_course(course_id: str, course_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return one course dict from the registry, or None if not found."""
    config = course_config if course_config is not None else load_course()
    for course in config.get("courses", []):
        if course.get("id") == course_id:
            return course
    return None


def get_term(term_id: str, glossary_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return one glossary term dict, or None if not found."""
    config = glossary_config if glossary_config is not None else load_glossary()
    for term in config.get("terms", []):
        if term.get("id") == term_id:
            return term
    return None


# -----------------------------------------------------------------------------
# Public-safe course payload
# -----------------------------------------------------------------------------


def course_public(course_id: str, course_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return a course payload with quizzes stripped of answers and explanations."""
    course = get_course(course_id, course_config=course_config)
    if course is None:
        return None

    public = copy.deepcopy(course)
    for module in public.get("modules", []):
        for quiz in module.get("quiz", []):
            quiz.pop("correct_index", None)
            quiz.pop("explanation", None)
    return public


def list_courses(course_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return a light list of courses (id, title, audience, description, module count, minutes)."""
    config = course_config if course_config is not None else load_course()
    result: List[Dict[str, Any]] = []
    for course in config.get("courses", []):
        modules = course.get("modules", [])
        result.append({
            "id": course.get("id"),
            "title": course.get("title"),
            "audience": course.get("audience"),
            "description": course.get("description"),
            "certificate_note": course.get("certificate_note"),
            "module_count": len(modules),
            "total_minutes": sum(int(m.get("minutes", 0)) for m in modules),
        })
    return result


# -----------------------------------------------------------------------------
# Grading
# -----------------------------------------------------------------------------


def _pass_threshold(num_questions: int) -> int:
    """Minimum correct answers to pass a module at 70%."""
    return math.ceil(0.7 * num_questions)


def grade(
    course_id: str,
    module_id: str,
    answers: List[int],
    course_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Grade a module quiz server-side.

    Returns a dict with score_correct, score_total, passed, and per-question
    results including the correct answer and explanation.
    """
    course = get_course(course_id, course_config=course_config)
    if course is None:
        raise AcademyConfigError(f"Unknown course '{course_id}'")

    module = None
    for m in course.get("modules", []):
        if m.get("id") == module_id:
            module = m
            break
    if module is None:
        raise AcademyConfigError(f"Unknown module '{module_id}' in course '{course_id}'")

    quiz = module.get("quiz") or []
    if len(answers) != len(quiz):
        raise ValueError(f"Expected {len(quiz)} answers, got {len(answers)}")

    correct_count = 0
    results: List[Dict[str, Any]] = []
    for idx, question in enumerate(quiz):
        correct_index = question.get("correct_index")
        user_answer = answers[idx]
        is_correct = correct_index is not None and user_answer == correct_index
        if is_correct:
            correct_count += 1
        results.append({
            "question": question.get("question"),
            "your_answer": user_answer,
            "correct_index": correct_index,
            "correct": is_correct,
            "explanation": question.get("explanation"),
        })

    total = len(quiz)
    threshold = _pass_threshold(total)
    passed = correct_count >= threshold

    return {
        "course_id": course_id,
        "module_id": module_id,
        "module_title": module.get("title"),
        "score_correct": correct_count,
        "score_total": total,
        "pass_threshold": threshold,
        "passed": passed,
        "results": results,
    }


# -----------------------------------------------------------------------------
# Knowledge graph (Academy 2.0)
# -----------------------------------------------------------------------------

# Rules-based mastery levels (Phase 1 of the Talaix learning engine; a
# Bayesian/Deep Knowledge Tracing model can replace these later once enough
# interaction data exists).
MASTERY_LEVELS = [
    ("mastered", 0.9),
    ("proficient", 0.8),
    ("developing", 0.6),
    ("needs_attention", 0.0),
]
PROFICIENT_LEVELS = {"mastered", "proficient"}

# SM-2-style spaced retrieval defaults.
REVIEW_INITIAL_INTERVAL = 3          # days until first review of a correct concept
REVIEW_FAIL_INTERVAL = 1             # days after a failed review / wrong answer
REVIEW_INITIAL_EASE = 2.5
REVIEW_MAX_EASE = 3.0
REVIEW_MIN_EASE = 1.3
REVIEW_EASE_GAIN = 0.15
REVIEW_EASE_DROP = 0.2
REVIEW_PASS_FRACTION = 0.7


def load_knowledge(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the academy knowledge graph registry."""
    path = path or os.environ.get("HYDRASHIELD_ACADEMY_KNOWLEDGE") or _DEFAULT_KNOWLEDGE_PATH
    data = _load_json(path, "Academy knowledge config")
    if not isinstance(data, dict) or "nodes" not in data:
        raise AcademyConfigError("Academy knowledge config must be an object with a 'nodes' list")
    return data


def get_knowledge(
    course_id: str,
    knowledge_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the knowledge graph for a course, or None if not found."""
    config = knowledge_config if knowledge_config is not None else load_knowledge()
    if config.get("course_id") == course_id:
        return config
    return None


def validate_knowledge(
    knowledge_config: Dict[str, Any],
    course: Dict[str, Any],
) -> List[str]:
    """
    Structural validation of a knowledge graph against its course.

    Returns a list of human-readable problems (empty when valid). Checks:
    unique node ids, resolvable prerequisites, module references that exist in
    the course, and competency concept references that exist as concept nodes.
    """
    problems: List[str] = []
    nodes = knowledge_config.get("nodes", [])
    node_ids = [n.get("id") for n in nodes]
    if len(node_ids) != len(set(node_ids)):
        problems.append("node ids must be unique")
    ids = set(node_ids)
    module_ids = {m.get("id") for m in course.get("modules", [])}
    module_by_id = {m.get("id"): m for m in course.get("modules", [])}

    for node in nodes:
        nid = node.get("id")
        if not nid:
            problems.append("node without id")
            continue
        if node.get("kind") not in ("root", "module", "concept"):
            problems.append(f"node '{nid}' has unknown kind {node.get('kind')!r}")
        for prereq in node.get("prerequisites", []):
            if prereq not in ids:
                problems.append(f"node '{nid}' prerequisite '{prereq}' does not exist")
        module_id = node.get("module_id")
        if module_id and module_id not in module_ids:
            problems.append(f"node '{nid}' references unknown module '{module_id}'")
        if module_id and module_id in module_by_id:
            # Every concept must be answerable by the module's quiz.
            if node.get("kind") == "concept":
                quiz = module_by_id[module_id].get("quiz", [])
                tagged = [q for q in quiz if nid in q.get("concepts", [])]
                if not tagged:
                    problems.append(f"concept '{nid}' has no tagged quiz question in module '{module_id}'")

    for competency in knowledge_config.get("competencies", []):
        for concept_id in competency.get("concepts", []):
            if concept_id not in ids:
                problems.append(
                    f"competency '{competency.get('id')}' references unknown concept '{concept_id}'"
                )

    return problems


def mastery_level(mastery: float) -> str:
    """Map a 0..1 mastery value to a rules-based level label."""
    for level, threshold in MASTERY_LEVELS:
        if mastery >= threshold:
            return level
    return "needs_attention"


def concept_mastery(attempts: List[Dict[str, Any]]) -> Tuple[float, int]:
    """
    Compute mastery for one concept from its attempt records.

    attempts is a list of dicts with a truthy 'correct' key. Returns
    (mastery 0..1, attempt_count). With no attempts, mastery is 0.
    """
    if not attempts:
        return 0.0, 0
    correct = sum(1 for a in attempts if a.get("correct"))
    return correct / len(attempts), len(attempts)


def _concept_level(mastery: float, attempts: int) -> str:
    if attempts == 0:
        return "not_started"
    return mastery_level(mastery)


def learner_model(
    knowledge_config: Dict[str, Any],
    attempts_by_concept: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Build the learner model from concept attempt records.

    Returns per-concept mastery (concepts and module hubs), competency
    mastery, weak areas, overall state and the recommended next step.
    """
    nodes = knowledge_config.get("nodes", [])
    by_id = {n["id"]: n for n in nodes}

    # Concept mastery from attempts.
    mastery: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        nid = node["id"]
        if node.get("kind") == "concept":
            m, attempts = concept_mastery(attempts_by_concept.get(nid, []))
            mastery[nid] = {
                "id": nid,
                "label": node.get("label"),
                "kind": node.get("kind"),
                "module_id": node.get("module_id"),
                "mastery": round(m, 4),
                "attempts": attempts,
                "level": _concept_level(m, attempts),
            }

    # Module hub mastery = mean of the module's concept mastery.
    module_children: Dict[str, List[str]] = {}
    for node in nodes:
        if node.get("kind") == "concept" and node.get("module_id"):
            module_children.setdefault(node["module_id"], []).append(node["id"])
    for node in nodes:
        if node.get("kind") != "module":
            continue
        children = module_children.get(node["module_id"], [])
        if not children:
            mastery[node["id"]] = {
                "id": node["id"], "label": node.get("label"), "kind": "module",
                "module_id": node.get("module_id"), "mastery": 0.0,
                "attempts": 0, "level": "not_started",
            }
            continue
        values = [mastery[c]["mastery"] for c in children if c in mastery]
        mean = sum(values) / len(values) if values else 0.0
        attempts = sum(mastery[c]["attempts"] for c in children if c in mastery)
        mastery[node["id"]] = {
            "id": node["id"], "label": node.get("label"), "kind": "module",
            "module_id": node.get("module_id"), "mastery": round(mean, 4),
            "attempts": attempts, "level": _concept_level(mean, attempts),
        }

    # Root = mean of all concept mastery.
    concepts = [n for n in nodes if n.get("kind") == "concept"]
    if concepts:
        mean = sum(mastery[c["id"]]["mastery"] for c in concepts) / len(concepts)
        mastery["course_root"] = {
            "id": "course_root", "label": knowledge_config.get("course_title"),
            "kind": "root", "module_id": None, "mastery": round(mean, 4),
            "attempts": sum(mastery[c["id"]]["attempts"] for c in concepts),
            "level": _concept_level(mean, sum(mastery[c["id"]]["attempts"] for c in concepts)),
        }

    # Competency mastery = mean of member concept mastery.
    competencies = []
    for comp in knowledge_config.get("competencies", []):
        member_ids = [c for c in comp.get("concepts", []) if c in mastery]
        values = [mastery[c]["mastery"] for c in member_ids] if member_ids else [0.0]
        mean = sum(values) / len(values)
        attempts = sum(mastery[c]["attempts"] for c in member_ids)
        competencies.append({
            "id": comp.get("id"),
            "label": comp.get("label"),
            "mastery": round(mean, 4),
            "attempts": attempts,
            "level": _concept_level(mean, attempts),
        })

    # Weak areas: concepts needing attention or still developing, weakest first.
    weak_areas = sorted(
        (
            mastery[n["id"]]
            for n in concepts
            if mastery[n["id"]]["level"] in ("needs_attention", "developing")
        ),
        key=lambda m: m["mastery"],
    )

    recommended = recommended_next(nodes, mastery)

    return {
        "concepts": [mastery[n["id"]] for n in nodes] + [mastery["course_root"]],
        "competencies": competencies,
        "weak_areas": [w["id"] for w in weak_areas],
        "recommended_next": recommended,
        "overall": mastery["course_root"],
    }


def recommended_next(
    nodes: List[Dict[str, Any]],
    mastery: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Rules-based adaptive recommendation: the first not-proficient concept in
    course order whose prerequisites are proficient — otherwise the earliest
    un-mastered prerequisite. Module hubs resolve to their first un-mastered
    concept so the recommendation is always actionable.
    """
    order = [n for n in nodes if n.get("kind") == "concept"]
    by_id = {n["id"]: n for n in nodes}

    def proficient(nid: str) -> bool:
        return mastery.get(nid, {}).get("level") in PROFICIENT_LEVELS

    for node in order:
        nid = node["id"]
        if proficient(nid):
            continue
        for prereq_id in node.get("prerequisites", []):
            prereq = by_id.get(prereq_id)
            if prereq is None or proficient(prereq_id):
                continue
            if prereq.get("kind") == "module":
                for child in order:
                    if (
                        child.get("module_id") == prereq.get("module_id")
                        and not proficient(child["id"])
                    ):
                        return {
                            "concept_id": child["id"],
                            "label": child.get("label"),
                            "reason": f"Master the prerequisite module '{prereq.get('label')}' first.",
                        }
                continue
            return {
                "concept_id": prereq_id,
                "label": prereq.get("label"),
                "reason": "Master the prerequisite first.",
            }
        return {
            "concept_id": nid,
            "label": node.get("label"),
            "reason": "Next concept on your learning path.",
        }
    return None


# -----------------------------------------------------------------------------
# Spaced retrieval (SM-2-lite)
# -----------------------------------------------------------------------------


def schedule_after_quiz(
    correct_by_concept: Dict[str, List[bool]],
    existing: Dict[str, Dict[str, Any]],
    now_ts: int,
) -> List[Dict[str, Any]]:
    """
    Update spaced-retrieval schedules after a module quiz.

    correct_by_concept maps concept_id -> list of booleans (one per tagged
    question in the submission). Wrong answers schedule an immediate review in
    REVIEW_FAIL_INTERVAL days; fully-correct concepts get REVIEW_INITIAL_INTERVAL
    days or an eased longer interval when a schedule already exists.
    """
    day = 24 * 60 * 60
    updated: List[Dict[str, Any]] = []
    for concept_id, results in correct_by_concept.items():
        if not results:
            continue
        all_correct = all(results)
        row = existing.get(concept_id)
        if all_correct:
            if row is None:
                interval = REVIEW_INITIAL_INTERVAL
                ease = REVIEW_INITIAL_EASE
            else:
                ease = min(REVIEW_MAX_EASE, float(row.get("ease", REVIEW_INITIAL_EASE)) + 0.1)
                interval = max(int(row.get("interval_days", REVIEW_INITIAL_INTERVAL)), 1)
                interval = max(interval, round(interval * ease))
            next_due = now_ts + interval * day
            updated.append({
                "concept_id": concept_id,
                "interval_days": interval,
                "ease": round(ease, 3),
                "next_due_ts": next_due,
                "last_result": 1,
            })
        else:
            updated.append({
                "concept_id": concept_id,
                "interval_days": REVIEW_FAIL_INTERVAL,
                "ease": round(REVIEW_INITIAL_EASE, 3),
                "next_due_ts": now_ts + REVIEW_FAIL_INTERVAL * day,
                "last_result": 0,
            })
    return updated


def apply_review(
    concept_id: str,
    results: List[bool],
    row: Dict[str, Any],
    now_ts: int,
) -> Dict[str, Any]:
    """
    Apply one spaced-retrieval session result (SM-2-lite).

    Passing (>= REVIEW_PASS_FRACTION correct) extends the interval by the ease
    factor; failing resets the interval and lowers ease.
    """
    day = 24 * 60 * 60
    if not results:
        passed = False
    else:
        passed = sum(1 for r in results if r) / len(results) >= REVIEW_PASS_FRACTION
    current_interval = int(row.get("interval_days", REVIEW_INITIAL_INTERVAL))
    current_ease = float(row.get("ease", REVIEW_INITIAL_EASE))

    if passed:
        interval = max(1, round(current_interval * current_ease))
        ease = min(REVIEW_MAX_EASE, current_ease + REVIEW_EASE_GAIN)
        next_due = now_ts + interval * day
    else:
        interval = REVIEW_FAIL_INTERVAL
        ease = max(REVIEW_MIN_EASE, current_ease - REVIEW_EASE_DROP)
        next_due = now_ts + interval * day

    return {
        "concept_id": concept_id,
        "interval_days": interval,
        "ease": round(ease, 3),
        "next_due_ts": next_due,
        "last_result": 1 if passed else 0,
        "passed": passed,
    }


def due_reviews(
    review_rows: List[Dict[str, Any]],
    now_ts: int,
) -> List[Dict[str, Any]]:
    """Filter review rows to those due at or before now_ts."""
    return [r for r in review_rows if int(r.get("next_due_ts", 0)) <= now_ts]


def review_questions_for_concept(
    course: Dict[str, Any],
    concept_id: str,
    concept: Dict[str, Any],
    include_answers: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return the module quiz questions tagged with a concept for a
    spaced-retrieval session. Answers (correct_index, explanation) are
    stripped by default; pass include_answers=True for server-side grading.
    """
    module_id = concept.get("module_id")
    module = next(
        (m for m in course.get("modules", []) if m.get("id") == module_id),
        None,
    )
    if module is None:
        return []
    questions = []
    for q in module.get("quiz", []):
        if concept_id not in q.get("concepts", []):
            continue
        if include_answers:
            questions.append(q)
            continue
        questions.append({
            "question": q.get("question"),
            "options": q.get("options", []),
        })
    return questions


def grade_review_answers(
    questions: List[Dict[str, Any]],
    answers: List[int],
) -> Tuple[int, int, List[bool]]:
    """Grade a spaced-retrieval session. Returns (correct, total, per-question bools)."""
    if len(answers) != len(questions):
        raise ValueError(f"Expected {len(questions)} answers, got {len(answers)}")
    results: List[bool] = []
    correct_count = 0
    for idx, q in enumerate(questions):
        is_correct = answers[idx] == q.get("correct_index")
        results.append(is_correct)
        if is_correct:
            correct_count += 1
    return correct_count, len(questions), results





def certificate_eligible(
    progress_rows: List[Dict[str, Any]],
    course: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Determine whether a user has passed all modules in a course.

    progress_rows are dicts with at least 'module_id' and 'passed' keys.
    Returns (eligible, missing_module_ids).
    """
    passed_modules = {r["module_id"] for r in progress_rows if r.get("passed")}
    required = [m["id"] for m in course.get("modules", [])]
    missing = [m for m in required if m not in passed_modules]
    return not missing, missing
