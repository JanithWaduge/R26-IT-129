from datetime import datetime, timezone
from typing import Any

from app.algorithms.mastery_update import (
    initial_direction_state,
)
from app.algorithms.review_priority import (
    calculate_review_priority,
)
from app.repositories.mastery_repository import (
    MasteryRepository,
)
from app.repositories.review_repository import (
    ReviewRepository,
)
from app.schemas.review import (
    DueReviewItemResponse,
    DueReviewListResponse,
    ReviewOverviewResponse,
)


class ReviewService:
    def __init__(
        self,
        *,
        review_repository: ReviewRepository,
        mastery_repository: MasteryRepository,
        environment: str,
    ) -> None:
        self.review_repository = (
            review_repository
        )

        self.mastery_repository = (
            mastery_repository
        )

        self.environment = environment

    def _visible_statuses(
        self,
    ) -> list[str]:
        if self.environment in {
            "development",
            "test",
        }:
            return [
                "approved",
                "development",
            ]

        return ["approved"]

    async def rank_candidates(
        self,
        *,
        student_id: str,
        signs: list[dict],
        mode: str,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        resolved_now = (
            now
            or datetime.now(timezone.utc)
        )

        sign_ids = [
            sign["_id"]
            for sign in signs
        ]

        category_ids = list(
            {
                sign["category_id"]
                for sign in signs
            }
        )

        competency_ids = list(
            {
                competency_id
                for sign in signs
                for competency_id
                in sign["competency_ids"]
            }
        )

        mastery_map = (
            await self.review_repository
            .get_mastery_map(
                student_id=student_id,
                sign_ids=sign_ids,
            )
        )

        category_weights = (
            await self.review_repository
            .get_category_weight_map(
                category_ids
            )
        )

        competency_weights = (
            await self.review_repository
            .get_competency_weight_map(
                competency_ids
            )
        )

        ranked: list[dict[str, Any]] = []

        for sign in signs:
            record = mastery_map.get(
                str(sign["_id"])
            )

            if record is None:
                receptive = (
                    initial_direction_state()
                )

                productive = (
                    initial_direction_state()
                )
            else:
                receptive = record[
                    "receptive"
                ]

                productive = record[
                    "productive"
                ]

            category_weight = (
                category_weights.get(
                    str(sign["category_id"]),
                    1.0,
                )
            )

            sign_competency_weights = [
                competency_weights.get(
                    str(competency_id),
                    1.0,
                )
                for competency_id
                in sign["competency_ids"]
            ]

            competency_weight = max(
                sign_competency_weights,
                default=1.0,
            )

            curriculum_weight = round(
                (
                    category_weight
                    + competency_weight
                )
                / 2,
                4,
            )

            allowed_directions = (
                ["receptive"]
                if mode == "receptive"
                else ["productive"]
                if mode == "productive"
                else [
                    "receptive",
                    "productive",
                ]
            )

            direction_candidates = []

            for direction in allowed_directions:
                if direction == "receptive":
                    state = receptive
                    opposite = productive
                else:
                    state = productive
                    opposite = receptive

                priority = (
                    calculate_review_priority(
                        direction_state=state,
                        opposite_state=opposite,
                        difficulty=sign[
                            "difficulty"
                        ],
                        curriculum_weight=(
                            curriculum_weight
                        ),
                        now=resolved_now,
                    )
                )

                direction_candidates.append(
                    {
                        "sign": sign,
                        "direction": direction,
                        "state": state,
                        "curriculum_weight": (
                            curriculum_weight
                        ),
                        "priority_score": (
                            priority.score
                        ),
                        "selection_reason": (
                            priority.reason
                        ),
                        "was_due": (
                            priority.is_due
                        ),
                        "days_overdue": (
                            priority.days_overdue
                        ),
                    }
                )

            if mode != "mixed":
                ranked.extend(
                    direction_candidates
                )
                continue

            direction_candidates.sort(
                key=lambda item: (
                    item["priority_score"],
                    -item["state"][
                        "total_reviews"
                    ],
                    (
                        1
                        if item["direction"]
                        == "productive"
                        else 0
                    ),
                ),
                reverse=True,
            )

            ranked.append(
                direction_candidates[0]
            )

        ranked.sort(
            key=lambda item: (
                item["priority_score"],
                item["days_overdue"],
                -item["state"][
                    "total_reviews"
                ],
            ),
            reverse=True,
        )

        return ranked

    async def get_due_reviews(
        self,
        *,
        student_id: str,
        mode: str,
        limit: int,
    ) -> DueReviewListResponse:
        signs = await (
            self.mastery_repository
            .list_visible_signs(
                visible_statuses=(
                    self._visible_statuses()
                )
            )
        )

        ranked = await self.rank_candidates(
            student_id=student_id,
            signs=signs,
            mode=mode,
        )

        due_items = [
            candidate
            for candidate in ranked
            if candidate["was_due"]
        ][:limit]

        return DueReviewListResponse(
            items=[
                self._to_due_item(
                    candidate
                )
                for candidate in due_items
            ],
            total_items=len(due_items),
            mode=mode,
        )

    async def get_overview(
        self,
        *,
        student_id: str,
    ) -> ReviewOverviewResponse:
        signs = await (
            self.mastery_repository
            .list_visible_signs(
                visible_statuses=(
                    self._visible_statuses()
                )
            )
        )

        receptive = await (
            self.rank_candidates(
                student_id=student_id,
                signs=signs,
                mode="receptive",
            )
        )

        productive = await (
            self.rank_candidates(
                student_id=student_id,
                signs=signs,
                mode="productive",
            )
        )

        all_candidates = [
            *receptive,
            *productive,
        ]

        due_candidates = [
            candidate
            for candidate in all_candidates
            if candidate["was_due"]
        ]

        new_receptive = sum(
            1
            for item in receptive
            if item["selection_reason"]
            == "new_direction"
        )

        new_productive = sum(
            1
            for item in productive
            if item["selection_reason"]
            == "new_direction"
        )

        overdue = sum(
            1
            for item in all_candidates
            if item["days_overdue"] >= 1
        )

        future_dates = [
            item["state"][
                "next_review_at"
            ]
            for item in all_candidates
            if (
                item["state"][
                    "next_review_at"
                ]
                is not None
                and not item["was_due"]
            )
        ]

        next_scheduled = (
            min(future_dates)
            if future_dates
            else None
        )

        return ReviewOverviewResponse(
            total_signs=len(signs),
            receptive_due=sum(
                1
                for item in receptive
                if item["was_due"]
            ),
            productive_due=sum(
                1
                for item in productive
                if item["was_due"]
            ),
            total_due_directions=len(
                due_candidates
            ),
            new_receptive_directions=(
                new_receptive
            ),
            new_productive_directions=(
                new_productive
            ),
            overdue_directions=overdue,
            next_scheduled_review_at=(
                next_scheduled
            ),
            highest_priority_score=max(
                (
                    item[
                        "priority_score"
                    ]
                    for item
                    in all_candidates
                ),
                default=0.0,
            ),
        )

    def _to_due_item(
        self,
        candidate: dict[str, Any],
    ) -> DueReviewItemResponse:
        sign = candidate["sign"]
        state = candidate["state"]

        return DueReviewItemResponse(
            sign_id=str(sign["_id"]),
            code=sign["code"],
            gloss=sign["gloss"],
            meanings=sign["meanings"],
            direction=candidate[
                "direction"
            ],
            difficulty=sign[
                "difficulty"
            ],
            mastery_score=float(
                state["score"]
            ),
            mastery_status=state[
                "status"
            ],
            ease_factor=float(
                state["ease_factor"]
            ),
            interval_days=state[
                "interval_days"
            ],
            repetition_count=state[
                "repetition_count"
            ],
            lapse_count=state[
                "lapse_count"
            ],
            next_review_at=state[
                "next_review_at"
            ],
            is_due=candidate[
                "was_due"
            ],
            days_overdue=candidate[
                "days_overdue"
            ],
            priority_score=candidate[
                "priority_score"
            ],
            selection_reason=candidate[
                "selection_reason"
            ],
            curriculum_weight=candidate[
                "curriculum_weight"
            ],
        )
