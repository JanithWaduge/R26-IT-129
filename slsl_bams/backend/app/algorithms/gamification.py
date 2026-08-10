from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BadgeDefinition:
    code: str
    title: str
    description: str
    icon_key: str


@dataclass(frozen=True)
class QuizRewardCalculation:
    total_xp: int
    question_xp: int
    due_review_bonus: int
    completion_bonus: int
    perfect_bonus: int
    is_perfect: bool


BADGE_DEFINITIONS: tuple[
    BadgeDefinition,
    ...,
] = (
    BadgeDefinition(
        code="first_quiz",
        title="First Step",
        description=(
            "Complete your first sign-video quiz."
        ),
        icon_key="flag",
    ),
    BadgeDefinition(
        code="five_quizzes",
        title="Quiz Explorer",
        description="Complete five quizzes.",
        icon_key="explore",
    ),
    BadgeDefinition(
        code="ten_quizzes",
        title="Dedicated Learner",
        description="Complete ten quizzes.",
        icon_key="school",
    ),
    BadgeDefinition(
        code="first_perfect_quiz",
        title="Perfect Round",
        description=(
            "Complete a quiz with every answer "
            "correct and without hints or retries."
        ),
        icon_key="stars",
    ),
    BadgeDefinition(
        code="ten_correct",
        title="Sign Reader",
        description=(
            "Answer ten receptive sign questions "
            "correctly."
        ),
        icon_key="visibility",
    ),
    BadgeDefinition(
        code="fifty_correct",
        title="Receptive Expert",
        description=(
            "Answer fifty receptive sign questions "
            "correctly."
        ),
        icon_key="workspace_premium",
    ),
    BadgeDefinition(
        code="streak_3",
        title="Three-Day Streak",
        description=(
            "Complete quizzes on three "
            "consecutive days."
        ),
        icon_key="local_fire_department",
    ),
    BadgeDefinition(
        code="streak_7",
        title="Seven-Day Streak",
        description=(
            "Complete quizzes on seven "
            "consecutive days."
        ),
        icon_key="whatshot",
    ),
    BadgeDefinition(
        code="level_5",
        title="Level Five",
        description="Reach learning level five.",
        icon_key="military_tech",
    ),
)


def calculate_level(
    *,
    total_xp: int,
    xp_per_level: int,
) -> tuple[int, int, int]:
    safe_xp = max(total_xp, 0)

    level = (
        safe_xp // xp_per_level
    ) + 1

    xp_into_level = (
        safe_xp % xp_per_level
    )

    return (
        level,
        xp_into_level,
        xp_per_level,
    )


def calculate_quiz_reward(
    session: dict[str, Any],
) -> QuizRewardCalculation:
    questions = session["questions"]

    question_xp = 0
    due_review_bonus = 0

    all_correct = len(questions) > 0
    no_assistance = True

    for question in questions:
        status = question["status"]

        if status == "correct":
            quality = question.get(
                "quality_score"
            )

            if quality is None:
                quality = 3.0

            quality_bonus = round(
                max(
                    0.0,
                    min(5.0, float(quality)),
                )
                * 2
            )

            question_xp += (
                10 + quality_bonus
            )

        elif status == "incorrect":
            question_xp += 3
            all_correct = False

        else:
            all_correct = False

        if question.get("was_due") is True:
            due_review_bonus += 2

        if (
            question.get("hint_count", 0) > 0
            or question.get(
                "retry_count",
                0,
            )
            > 0
        ):
            no_assistance = False

    completion_bonus = 10

    is_perfect = (
        all_correct
        and no_assistance
        and session["summary"][
            "skipped_count"
        ]
        == 0
    )

    perfect_bonus = (
        20 if is_perfect else 0
    )

    total_xp = (
        question_xp
        + due_review_bonus
        + completion_bonus
        + perfect_bonus
    )

    return QuizRewardCalculation(
        total_xp=total_xp,
        question_xp=question_xp,
        due_review_bonus=(
            due_review_bonus
        ),
        completion_bonus=(
            completion_bonus
        ),
        perfect_bonus=perfect_bonus,
        is_perfect=is_perfect,
    )


def determine_earned_badges(
    *,
    total_quizzes: int,
    total_correct: int,
    perfect_quizzes: int,
    current_streak_days: int,
    level: int,
) -> set[str]:
    earned: set[str] = set()

    if total_quizzes >= 1:
        earned.add("first_quiz")

    if total_quizzes >= 5:
        earned.add("five_quizzes")

    if total_quizzes >= 10:
        earned.add("ten_quizzes")

    if perfect_quizzes >= 1:
        earned.add(
            "first_perfect_quiz"
        )

    if total_correct >= 10:
        earned.add("ten_correct")

    if total_correct >= 50:
        earned.add("fifty_correct")

    if current_streak_days >= 3:
        earned.add("streak_3")

    if current_streak_days >= 7:
        earned.add("streak_7")

    if level >= 5:
        earned.add("level_5")

    return earned


def badge_definition(
    code: str,
) -> BadgeDefinition:
    for definition in BADGE_DEFINITIONS:
        if definition.code == code:
            return definition

    return BadgeDefinition(
        code=code,
        title=code.replace(
            "_",
            " ",
        ).title(),
        description="Achievement unlocked.",
        icon_key="emoji_events",
    )