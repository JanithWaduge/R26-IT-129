from copy import deepcopy
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
from zoneinfo import ZoneInfo

from bson import ObjectId

from app.algorithms.gamification import (
    BADGE_DEFINITIONS,
    badge_definition,
    calculate_level,
    calculate_quiz_reward,
    determine_earned_badges,
)
from app.core.config import Settings
from app.repositories.gamification_repository import (
    GamificationRepository,
)
from app.schemas.gamification import (
    GamificationBadgeResponse,
    GamificationHistoryItemResponse,
    GamificationHistoryResponse,
    GamificationProfileResponse,
    GamificationRewardBreakdownResponse,
    GamificationRewardResponse,
)


class GamificationService:
    def __init__(
        self,
        *,
        repository: GamificationRepository,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.settings = settings

    async def apply_completed_session(
        self,
        session: dict,
    ) -> GamificationRewardResponse:
        if session["status"] != "completed":
            raise ValueError(
                "Only completed quizzes can "
                "receive rewards."
            )

        student_id: ObjectId = session[
            "student_id"
        ]

        session_id: ObjectId = session[
            "_id"
        ]

        reward = calculate_quiz_reward(
            session
        )

        now = datetime.now(timezone.utc)

        event = await (
            self.repository
            .create_or_get_event(
                {
                    "student_id": student_id,
                    "session_id": session_id,
                    "status": "pending",
                    "xp_awarded": (
                        reward.total_xp
                    ),
                    "badges_awarded": [],
                    "reward_breakdown": {
                        "question_xp": (
                            reward.question_xp
                        ),
                        "due_review_bonus": (
                            reward
                            .due_review_bonus
                        ),
                        "completion_bonus": (
                            reward
                            .completion_bonus
                        ),
                        "perfect_bonus": (
                            reward.perfect_bonus
                        ),
                    },
                    "total_xp_after": None,
                    "level_after": None,
                    "streak_after": None,
                    "created_at": now,
                    "applied_at": None,
                }
            )
        )

        if event["status"] == "applied":
            return self._event_to_reward(
                event
            )

        for _ in range(8):
            profile = await (
                self.repository
                .ensure_profile(
                    student_id=student_id,
                    xp_per_level=(
                        self.settings
                        .gamification_xp_per_level
                    ),
                )
            )

            if session_id in profile[
                "processed_session_ids"
            ]:
                repaired_event = await (
                    self.repository
                    .mark_event_applied(
                        session_id=session_id,
                        badges_awarded=event[
                            "badges_awarded"
                        ],
                        total_xp_after=profile[
                            "total_xp"
                        ],
                        level_after=profile[
                            "level"
                        ],
                        streak_after=profile[
                            "current_streak_days"
                        ],
                        applied_at=now,
                    )
                )

                return self._event_to_reward(
                    repaired_event
                )

            replacement = deepcopy(
                profile
            )

            current_date = self._today()
            current_date_text = (
                current_date.isoformat()
            )

            streak = self._updated_streak(
                previous_date_text=profile[
                    "last_activity_date"
                ],
                previous_streak=profile[
                    "current_streak_days"
                ],
                current_date=current_date,
            )

            replacement[
                "current_streak_days"
            ] = streak

            replacement[
                "longest_streak_days"
            ] = max(
                profile[
                    "longest_streak_days"
                ],
                streak,
            )

            replacement[
                "last_activity_date"
            ] = current_date_text

            replacement["total_xp"] += (
                reward.total_xp
            )

            replacement[
                "total_quizzes"
            ] += 1

            replacement[
                "total_questions"
            ] += session[
                "question_count"
            ]

            replacement[
                "total_correct"
            ] += session["summary"][
                "correct_count"
            ]

            replacement[
                "total_incorrect"
            ] += session["summary"][
                "incorrect_count"
            ]

            replacement[
                "total_skipped"
            ] += session["summary"][
                "skipped_count"
            ]

            replacement[
                "total_hints"
            ] += session["summary"][
                "hints_used"
            ]

            replacement[
                "total_retries"
            ] += session["summary"][
                "total_retries"
            ]

            if reward.is_perfect:
                replacement[
                    "perfect_quizzes"
                ] += 1

            (
                level,
                xp_into_level,
                xp_to_next,
            ) = calculate_level(
                total_xp=replacement[
                    "total_xp"
                ],
                xp_per_level=(
                    self.settings
                    .gamification_xp_per_level
                ),
            )

            replacement["level"] = level
            replacement[
                "xp_into_level"
            ] = xp_into_level

            replacement[
                "xp_to_next_level"
            ] = xp_to_next

            existing_codes = {
                badge["code"]
                for badge in replacement[
                    "unlocked_badges"
                ]
            }

            earned_codes = (
                determine_earned_badges(
                    total_quizzes=replacement[
                        "total_quizzes"
                    ],
                    total_correct=replacement[
                        "total_correct"
                    ],
                    perfect_quizzes=(
                        replacement[
                            "perfect_quizzes"
                        ]
                    ),
                    current_streak_days=(
                        replacement[
                            "current_streak_days"
                        ]
                    ),
                    level=level,
                )
            )

            new_codes = sorted(
                earned_codes
                - existing_codes
            )

            for code in new_codes:
                replacement[
                    "unlocked_badges"
                ].append(
                    {
                        "code": code,
                        "unlocked_at": now,
                        "source_session_id": (
                            session_id
                        ),
                    }
                )

            replacement[
                "processed_session_ids"
            ].append(session_id)

            replacement["updated_at"] = now

            updated = await (
                self.repository
                .replace_profile(
                    profile=replacement,
                    expected_version=profile[
                        "version"
                    ],
                    session_id=session_id,
                )
            )

            if updated is None:
                continue

            applied_event = await (
                self.repository
                .mark_event_applied(
                    session_id=session_id,
                    badges_awarded=(
                        new_codes
                    ),
                    total_xp_after=updated[
                        "total_xp"
                    ],
                    level_after=updated[
                        "level"
                    ],
                    streak_after=updated[
                        "current_streak_days"
                    ],
                    applied_at=now,
                )
            )

            return self._event_to_reward(
                applied_event
            )

        raise RuntimeError(
            "The quiz reward could not be "
            "applied after multiple attempts."
        )

    async def get_profile(
        self,
        *,
        student_id: str,
    ) -> GamificationProfileResponse:
        object_id = (
            self.repository.parse_object_id(
                student_id
            )
        )

        profile = await (
            self.repository.ensure_profile(
                student_id=object_id,
                xp_per_level=(
                    self.settings
                    .gamification_xp_per_level
                ),
            )
        )

        return self._profile_to_response(
            profile
        )

    async def get_history(
        self,
        *,
        student_id: str,
        limit: int,
    ) -> GamificationHistoryResponse:
        object_id = (
            self.repository.parse_object_id(
                student_id
            )
        )

        events = await (
            self.repository.list_events(
                student_id=object_id,
                limit=limit,
            )
        )

        return GamificationHistoryResponse(
            items=[
                GamificationHistoryItemResponse(
                    session_id=str(
                        event["session_id"]
                    ),
                    xp_awarded=event[
                        "xp_awarded"
                    ],
                    badges_awarded=event[
                        "badges_awarded"
                    ],
                    total_xp_after=event[
                        "total_xp_after"
                    ],
                    level_after=event[
                        "level_after"
                    ],
                    streak_after=event[
                        "streak_after"
                    ],
                    created_at=event[
                        "created_at"
                    ],
                )
                for event in events
            ]
        )

    def _today(self) -> date:
        timezone_value = ZoneInfo(
            self.settings
            .gamification_timezone
        )

        return datetime.now(
            timezone_value
        ).date()

    def _updated_streak(
        self,
        *,
        previous_date_text: str | None,
        previous_streak: int,
        current_date: date,
    ) -> int:
        if previous_date_text is None:
            return 1

        previous_date = date.fromisoformat(
            previous_date_text
        )

        if previous_date == current_date:
            return max(previous_streak, 1)

        if (
            previous_date
            == current_date
            - timedelta(days=1)
        ):
            return previous_streak + 1

        return 1

    def _profile_to_response(
        self,
        profile: dict,
    ) -> GamificationProfileResponse:
        unlocked_map = {
            badge["code"]: badge
            for badge in profile[
                "unlocked_badges"
            ]
        }

        badges = []

        for definition in BADGE_DEFINITIONS:
            unlocked = unlocked_map.get(
                definition.code
            )

            badges.append(
                GamificationBadgeResponse(
                    code=definition.code,
                    title=definition.title,
                    description=(
                        definition.description
                    ),
                    icon_key=(
                        definition.icon_key
                    ),
                    unlocked=(
                        unlocked is not None
                    ),
                    unlocked_at=(
                        unlocked[
                            "unlocked_at"
                        ]
                        if unlocked
                        else None
                    ),
                )
            )

        answered = (
            profile["total_correct"]
            + profile["total_incorrect"]
        )

        accuracy = (
            0.0
            if answered == 0
            else round(
                (
                    profile[
                        "total_correct"
                    ]
                    / answered
                )
                * 100,
                2,
            )
        )

        return GamificationProfileResponse(
            total_xp=profile["total_xp"],
            level=profile["level"],
            xp_into_level=profile[
                "xp_into_level"
            ],
            xp_to_next_level=profile[
                "xp_to_next_level"
            ],
            current_streak_days=profile[
                "current_streak_days"
            ],
            longest_streak_days=profile[
                "longest_streak_days"
            ],
            last_activity_date=profile[
                "last_activity_date"
            ],
            total_quizzes=profile[
                "total_quizzes"
            ],
            total_questions=profile[
                "total_questions"
            ],
            total_correct=profile[
                "total_correct"
            ],
            total_incorrect=profile[
                "total_incorrect"
            ],
            total_skipped=profile[
                "total_skipped"
            ],
            total_hints=profile[
                "total_hints"
            ],
            total_retries=profile[
                "total_retries"
            ],
            perfect_quizzes=profile[
                "perfect_quizzes"
            ],
            accuracy_percentage=accuracy,
            badges=badges,
        )

    def _event_to_reward(
        self,
        event: dict,
    ) -> GamificationRewardResponse:
        badges = []

        for code in event[
            "badges_awarded"
        ]:
            definition = badge_definition(
                code
            )

            badges.append(
                GamificationBadgeResponse(
                    code=definition.code,
                    title=definition.title,
                    description=(
                        definition.description
                    ),
                    icon_key=(
                        definition.icon_key
                    ),
                    unlocked=True,
                    unlocked_at=event[
                        "applied_at"
                    ],
                )
            )

        (
            _,
            xp_into_level,
            xp_to_next_level,
        ) = calculate_level(
            total_xp=event[
                "total_xp_after"
            ],
            xp_per_level=(
                self.settings
                .gamification_xp_per_level
            ),
        )

        return GamificationRewardResponse(
            xp_awarded=event[
                "xp_awarded"
            ],
            total_xp=event[
                "total_xp_after"
            ],
            level=event["level_after"],
            xp_into_level=xp_into_level,
            xp_to_next_level=(
                xp_to_next_level
            ),
            current_streak_days=event[
                "streak_after"
            ],
            badges_awarded=badges,
            breakdown=(
                GamificationRewardBreakdownResponse(
                    **event[
                        "reward_breakdown"
                    ]
                )
            ),
        )