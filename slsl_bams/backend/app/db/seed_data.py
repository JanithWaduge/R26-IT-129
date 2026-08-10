from datetime import datetime, timezone
from typing import Any

from bson import ObjectId


CATEGORY_IDS = {
    "greetings": ObjectId(
        "65a000000000000000000001"
    ),
    "school": ObjectId(
        "65a000000000000000000002"
    ),
    "numbers": ObjectId(
        "65a000000000000000000003"
    ),
    "emotions": ObjectId(
        "65a000000000000000000004"
    ),
    "daily_life": ObjectId(
        "65a000000000000000000005"
    ),
}


COMPETENCY_IDS = {
    "greetings_basic": ObjectId(
        "65b000000000000000000001"
    ),
    "school_basic": ObjectId(
        "65b000000000000000000002"
    ),
    "numbers_basic": ObjectId(
        "65b000000000000000000003"
    ),
    "emotions_basic": ObjectId(
        "65b000000000000000000004"
    ),
    "daily_life_basic": ObjectId(
        "65b000000000000000000005"
    ),
}


SIGN_IDS = {
    "hello": ObjectId(
        "65c000000000000000000001"
    ),
    "school": ObjectId(
        "65c000000000000000000002"
    ),
    "one": ObjectId(
        "65c000000000000000000003"
    ),
    "happy": ObjectId(
        "65c000000000000000000004"
    ),
    "eat": ObjectId(
        "65c000000000000000000005"
    ),
}


def build_seed_documents() -> dict[str, list[dict[str, Any]]]:
    now = datetime.now(timezone.utc)

    categories = [
        {
            "_id": CATEGORY_IDS["greetings"],
            "code": "GREETINGS",
            "name": {
                "english": "Greetings",
                "sinhala": "ආචාර",
                "tamil": "வாழ்த்துகள்",
            },
            "description": {
                "english": (
                    "Development category for common "
                    "greeting concepts."
                ),
                "sinhala": (
                    "සාමාන්‍ය ආචාර සංකල්ප සඳහා "
                    "සංවර්ධන කාණ්ඩයකි."
                ),
                "tamil": (
                    "பொதுவான வாழ்த்து கருத்துகளுக்கான "
                    "மேம்பாட்டு வகை."
                ),
            },
            "display_order": 1,
            "icon_key": "waving_hand",
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": CATEGORY_IDS["school"],
            "code": "SCHOOL_VOCABULARY",
            "name": {
                "english": "School Vocabulary",
                "sinhala": "පාසල් වචන මාලාව",
                "tamil": "பாடசாலை சொற்கள்",
            },
            "description": {
                "english": (
                    "Development category for school "
                    "and learning concepts."
                ),
                "sinhala": (
                    "පාසල් සහ ඉගෙනුම් සංකල්ප සඳහා "
                    "සංවර්ධන කාණ්ඩයකි."
                ),
                "tamil": (
                    "பாடசாலை மற்றும் கற்றல் "
                    "கருத்துகளுக்கான மேம்பாட்டு வகை."
                ),
            },
            "display_order": 2,
            "icon_key": "school",
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": CATEGORY_IDS["numbers"],
            "code": "NUMBERS",
            "name": {
                "english": "Numbers",
                "sinhala": "අංක",
                "tamil": "எண்கள்",
            },
            "description": {
                "english": (
                    "Development category for basic "
                    "number concepts."
                ),
                "sinhala": (
                    "මූලික අංක සංකල්ප සඳහා "
                    "සංවර්ධන කාණ්ඩයකි."
                ),
                "tamil": (
                    "அடிப்படை எண் கருத்துகளுக்கான "
                    "மேம்பாட்டு வகை."
                ),
            },
            "display_order": 3,
            "icon_key": "numbers",
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": CATEGORY_IDS["emotions"],
            "code": "EMOTIONS",
            "name": {
                "english": "Emotions",
                "sinhala": "හැඟීම්",
                "tamil": "உணர்வுகள்",
            },
            "description": {
                "english": (
                    "Development category for basic "
                    "emotion concepts."
                ),
                "sinhala": (
                    "මූලික හැඟීම් සංකල්ප සඳහා "
                    "සංවර්ධන කාණ්ඩයකි."
                ),
                "tamil": (
                    "அடிப்படை உணர்வு கருத்துகளுக்கான "
                    "மேம்பாட்டு வகை."
                ),
            },
            "display_order": 4,
            "icon_key": "mood",
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": CATEGORY_IDS["daily_life"],
            "code": "DAILY_LIFE",
            "name": {
                "english": "Daily Life",
                "sinhala": "දෛනික ජීවිතය",
                "tamil": "அன்றாட வாழ்க்கை",
            },
            "description": {
                "english": (
                    "Development category for common "
                    "daily activities."
                ),
                "sinhala": (
                    "සාමාන්‍ය දෛනික ක්‍රියාකාරකම් සඳහා "
                    "සංවර්ධන කාණ්ඩයකි."
                ),
                "tamil": (
                    "பொதுவான அன்றாட செயல்பாடுகளுக்கான "
                    "மேம்பாட்டு வகை."
                ),
            },
            "display_order": 5,
            "icon_key": "daily_activity",
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
    ]

    competencies = [
        {
            "_id": COMPETENCY_IDS["greetings_basic"],
            "category_id": CATEGORY_IDS["greetings"],
            "code": "DEV-GREET-01",
            "title": {
                "english": "Identify basic greetings",
                "sinhala": "මූලික ආචාර හඳුනාගැනීම",
                "tamil": "அடிப்படை வாழ்த்துகளை அடையாளம் காணுதல்",
            },
            "description": {
                "english": (
                    "Provisional development competency; "
                    "teacher validation is required."
                ),
                "sinhala": (
                    "තාවකාලික සංවර්ධන නිපුණතාවකි; "
                    "ගුරු අනුමැතිය අවශ්‍ය වේ."
                ),
                "tamil": (
                    "தற்காலிக மேம்பாட்டு திறன்; "
                    "ஆசிரியர் சரிபார்ப்பு தேவை."
                ),
            },
            "grade_levels": ["Development"],
            "display_order": 1,
            "receptive_mastery_threshold": 0.85,
            "productive_mastery_threshold": 0.85,
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": COMPETENCY_IDS["school_basic"],
            "category_id": CATEGORY_IDS["school"],
            "code": "DEV-SCHOOL-01",
            "title": {
                "english": "Identify basic school concepts",
                "sinhala": "මූලික පාසල් සංකල්ප හඳුනාගැනීම",
                "tamil": "அடிப்படை பாடசாலை கருத்துகளை அடையாளம் காணுதல்",
            },
            "description": {
                "english": (
                    "Provisional development competency; "
                    "teacher validation is required."
                ),
                "sinhala": (
                    "තාවකාලික සංවර්ධන නිපුණතාවකි; "
                    "ගුරු අනුමැතිය අවශ්‍ය වේ."
                ),
                "tamil": (
                    "தற்காலிக மேம்பாட்டு திறன்; "
                    "ஆசிரியர் சரிபார்ப்பு தேவை."
                ),
            },
            "grade_levels": ["Development"],
            "display_order": 1,
            "receptive_mastery_threshold": 0.85,
            "productive_mastery_threshold": 0.85,
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": COMPETENCY_IDS["numbers_basic"],
            "category_id": CATEGORY_IDS["numbers"],
            "code": "DEV-NUMBER-01",
            "title": {
                "english": "Identify basic numbers",
                "sinhala": "මූලික අංක හඳුනාගැනීම",
                "tamil": "அடிப்படை எண்களை அடையாளம் காணுதல்",
            },
            "description": {
                "english": (
                    "Provisional development competency; "
                    "teacher validation is required."
                ),
                "sinhala": (
                    "තාවකාලික සංවර්ධන නිපුණතාවකි; "
                    "ගුරු අනුමැතිය අවශ්‍ය වේ."
                ),
                "tamil": (
                    "தற்காலிக மேம்பாட்டு திறன்; "
                    "ஆசிரியர் சரிபார்ப்பு தேவை."
                ),
            },
            "grade_levels": ["Development"],
            "display_order": 1,
            "receptive_mastery_threshold": 0.85,
            "productive_mastery_threshold": 0.85,
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": COMPETENCY_IDS["emotions_basic"],
            "category_id": CATEGORY_IDS["emotions"],
            "code": "DEV-EMOTION-01",
            "title": {
                "english": "Identify basic emotions",
                "sinhala": "මූලික හැඟීම් හඳුනාගැනීම",
                "tamil": "அடிப்படை உணர்வுகளை அடையாளம் காணுதல்",
            },
            "description": {
                "english": (
                    "Provisional development competency; "
                    "teacher validation is required."
                ),
                "sinhala": (
                    "තාවකාලික සංවර්ධන නිපුණතාවකි; "
                    "ගුරු අනුමැතිය අවශ්‍ය වේ."
                ),
                "tamil": (
                    "தற்காலிக மேம்பாட்டு திறன்; "
                    "ஆசிரியர் சரிபார்ப்பு தேவை."
                ),
            },
            "grade_levels": ["Development"],
            "display_order": 1,
            "receptive_mastery_threshold": 0.85,
            "productive_mastery_threshold": 0.85,
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "_id": COMPETENCY_IDS["daily_life_basic"],
            "category_id": CATEGORY_IDS["daily_life"],
            "code": "DEV-DAILY-01",
            "title": {
                "english": "Identify basic daily activities",
                "sinhala": "මූලික දෛනික ක්‍රියා හඳුනාගැනීම",
                "tamil": "அடிப்படை அன்றாட செயல்களை அடையாளம் காணுதல்",
            },
            "description": {
                "english": (
                    "Provisional development competency; "
                    "teacher validation is required."
                ),
                "sinhala": (
                    "තාවකාලික සංවර්ධන නිපුණතාවකි; "
                    "ගුරු අනුමැතිය අවශ්‍ය වේ."
                ),
                "tamil": (
                    "தற்காலிக மேம்பாட்டு திறன்; "
                    "ஆசிரியர் சரிபார்ப்பு தேவை."
                ),
            },
            "grade_levels": ["Development"],
            "display_order": 1,
            "receptive_mastery_threshold": 0.85,
            "productive_mastery_threshold": 0.85,
            "adaptive_priority_weight": 1.0,
            "validation_status": "provisional",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        },
    ]

    signs = [
_build_sign(
    now=now,
    sign_id=SIGN_IDS["hello"],
    code="SLSL-DEMO-HELLO",
    gloss="HELLO",
    english="Hello",
    sinhala="ආයුබෝවන්",
    tamil="வணக்கம்",
    category_id=CATEGORY_IDS[
        "greetings"
    ],
    competency_id=COMPETENCY_IDS[
        "greetings_basic"
    ],
    difficulty=1,
    tags=["greeting", "basic"],
    video_filename="Ability.mp4",
    duration_ms=3000,
),
        _build_sign(
            now=now,
            sign_id=SIGN_IDS["school"],
            code="SLSL-DEMO-SCHOOL",
            gloss="SCHOOL",
            english="School",
            sinhala="පාසල",
            tamil="பாடசாலை",
            category_id=CATEGORY_IDS["school"],
            competency_id=COMPETENCY_IDS["school_basic"],
            difficulty=1,
            tags=["school", "education"],
            video_filename="Academic Advisor.mp4",
            duration_ms=3000,
        ),
        _build_sign(
            now=now,
            sign_id=SIGN_IDS["one"],
            code="SLSL-DEMO-ONE",
            gloss="ONE",
            english="One",
            sinhala="එක",
            tamil="ஒன்று",
            category_id=CATEGORY_IDS["numbers"],
            competency_id=COMPETENCY_IDS["numbers_basic"],
            difficulty=1,
            tags=["number", "basic"],
            video_filename="Academic Year.mp4",
            duration_ms=2500,
        ),
        _build_sign(
            now=now,
            sign_id=SIGN_IDS["happy"],
            code="SLSL-DEMO-HAPPY",
            gloss="HAPPY",
            english="Happy",
            sinhala="සතුටු",
            tamil="மகிழ்ச்சி",
            category_id=CATEGORY_IDS["emotions"],
            competency_id=COMPETENCY_IDS["emotions_basic"],
            difficulty=2,
            tags=["emotion", "feeling"],
            video_filename="Activity.mp4",
            duration_ms=3000,
        ),
        _build_sign(
            now=now,
            sign_id=SIGN_IDS["eat"],
            code="SLSL-DEMO-EAT",
            gloss="EAT",
            english="Eat",
            sinhala="කන්න",
            tamil="சாப்பிடு",
            category_id=CATEGORY_IDS["daily_life"],
            competency_id=COMPETENCY_IDS["daily_life_basic"],
            difficulty=2,
            tags=["daily-life", "activity"],
            video_filename="Answer Sheet.mp4",
            duration_ms=3000,
        ),
    ]

    return {
        "categories": categories,
        "competencies": competencies,
        "signs": signs,
    }


def _build_sign(
    *,
    now: datetime,
    sign_id: ObjectId,
    code: str,
    gloss: str,
    english: str,
    sinhala: str,
    tamil: str,
    category_id: ObjectId,
    competency_id: ObjectId,
    difficulty: int,
    tags: list[str],
    video_filename: str,
    duration_ms: int,
) -> dict[str, Any]:
    search_terms = sorted(
        {
            code.lower(),
            gloss.lower(),
            english.lower(),
            sinhala,
            tamil,
            *[
                tag.lower()
                for tag in tags
            ],
        }
    )

    return {
        "_id": sign_id,
        "code": code,
        "gloss": gloss,
        "meanings": {
            "english": english,
            "sinhala": sinhala,
            "tamil": tamil,
        },
        "category_id": category_id,
        "competency_ids": [
            competency_id
        ],
        "difficulty": difficulty,
        "tags": tags,
        "search_terms": search_terms,
        "prerequisite_sign_ids": [],
    "media": {
    "source_type": "video",
    "uri": (
        "/media/sign-videos/"
        f"{video_filename}"
    ),
    "thumbnail_uri": None,
    "duration_ms": duration_ms,
    "objective_source": (
        "teacher_upload"
    ),
},
        "content_status": "development",
        "validation_status": "provisional",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
