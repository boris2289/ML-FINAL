from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    student_target: str = Field(..., description="Ожидаемый балл студента (строка)")
    student_class: str = Field(..., description="Класс студента: 9, 10 или 11")
    course_name: str = Field(..., description="Название курса")
    subject_name: str = Field(..., description="Название предмета")

    homework_done_mark: float = Field(..., description="Средний балл за ДЗ")
    test_part_one: float = Field(..., description="Средний балл за тест (часть 1)")
    test_part_two: float = Field(..., description="Средний балл за тест (часть 2)")

    homework_lag_1: float = Field(-1, description="Балл за ДЗ на прошлом уроке")
    homework_lag_2: float = Field(-1, description="Балл за ДЗ 2 урока назад")
    test1_lag_1: float = Field(-1, description="Тест ч.1 на прошлом уроке")
    test2_lag_1: float = Field(-1, description="Тест ч.2 на прошлом уроке")

    homework_diff: float = Field(0, description="Изменение балла ДЗ")
    test1_diff: float = Field(0, description="Изменение балла теста ч.1")
    test2_diff: float = Field(0, description="Изменение балла теста ч.2")

    homework_rolling_mean_3: float = Field(-1, description="Скользящее среднее ДЗ за 3 урока")
    homework_rolling_std_3: float = Field(-1, description="Скользящее отклонение ДЗ за 3 урока")
    test1_rolling_mean_3: float = Field(-1, description="Скользящее среднее теста ч.1 за 3 урока")
    test2_rolling_std_3: float = Field(-1, description="Скользящее отклонение теста ч.2 за 3 урока")

    homework_max: float = Field(0, description="Максимальный балл за ДЗ")
    homework_min: float = Field(0, description="Минимальный балл за ДЗ")
    test1_max: float = Field(0, description="Максимальный балл за тест ч.1")
    test1_min: float = Field(0, description="Минимальный балл за тест ч.1")
    test2_max: float = Field(0, description="Максимальный балл за тест ч.2")
    test2_min: float = Field(0, description="Минимальный балл за тест ч.2")

    @field_validator("student_class")
    @classmethod
    def validate_class(cls, value: str) -> str:
        if value not in ("9", "10", "11"):
            raise ValueError("student_class must be '9', '10', or '11'.")
        return value


class PredictionResponse(BaseModel):
    predicted_ege_score: float
