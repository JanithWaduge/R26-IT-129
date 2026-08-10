from fastapi import APIRouter

from app.api.routes import (
    auth,
    curriculum,
    gamification,
    health,
    mastery,
    ml,
    quizzes,
    recognition,
    reviews,
    signs,
    students,
)

api_router = APIRouter()

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    students.router,
    prefix="/students",
    tags=["Students"],
)

api_router.include_router(
    curriculum.router,
    prefix="/curriculum",
    tags=["Curriculum"],
)

api_router.include_router(
    signs.router,
    prefix="/signs",
    tags=["Signs"],
)

api_router.include_router(
    quizzes.router,
    prefix="/quizzes",
    tags=["Quizzes"],
)

api_router.include_router(
    mastery.router,
    prefix="/mastery",
    tags=["Mastery"],
)

api_router.include_router(
    ml.router,
    prefix="/ml",
    tags=["Machine Learning"],
)

api_router.include_router(
    reviews.router,
    prefix="/reviews",
    tags=["Reviews"],
)

api_router.include_router(
    gamification.router,
    prefix="/gamification",
    tags=["Gamification"],
)

api_router.include_router(
    recognition.router,
    prefix="/quizzes",
    tags=["Recognition"],
)
