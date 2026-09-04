"""Human-readable Russian labels for the model's technical class keys."""

from __future__ import annotations

LABELS: dict[str, dict[str, str]] = {
    "ampulla_of_vater": {
        "display_name": "Большой дуоденальный сосочек",
        "category": "анатомический ориентир",
    },
    "angiectasia": {"display_name": "Ангиоэктазия", "category": "находка"},
    "blood_fresh": {"display_name": "Свежая кровь", "category": "находка"},
    "blood_hematin": {
        "display_name": "Гематин / изменённая кровь",
        "category": "находка",
    },
    "erosion": {"display_name": "Эрозия", "category": "находка"},
    "erythema": {"display_name": "Эритема", "category": "находка"},
    "foreign_body": {"display_name": "Инородное тело", "category": "находка"},
    "ileocecal_valve": {
        "display_name": "Илеоцекальный клапан",
        "category": "анатомический ориентир",
    },
    "lymphangiectasia": {"display_name": "Лимфангиэктазия", "category": "находка"},
    "normal_clean_mucosa": {
        "display_name": "Нормальная чистая слизистая",
        "category": "норма",
    },
    "polyp": {"display_name": "Полип", "category": "находка"},
    "pylorus": {
        "display_name": "Привратник",
        "category": "анатомический ориентир",
    },
    "reduced_mucosal_view": {
        "display_name": "Ограниченный обзор слизистой",
        "category": "качество изображения",
    },
    "ulcer": {"display_name": "Язва", "category": "находка"},
}


def describe_label(class_key: str) -> dict[str, str]:
    """Return safe presentation metadata, preserving unknown checkpoint keys."""
    known = LABELS.get(class_key)
    if known is None:
        return {
            "class_key": class_key,
            "display_name": class_key,
            "category": "класс модели",
        }
    return {"class_key": class_key, **known}
