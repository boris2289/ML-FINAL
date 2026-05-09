SUBJECT_NAMES = [
    "Обществознание",
    "История",
    "Литература",
    "Русский",
    "Английский язык",
    "Математика",
    "Физика",
    "Биология",
    "Химия",
    "Информатика",
    "Русский ОГЭ",
    "Математика ОГЭ",
    "Обществознание ОГЭ",
    "Биология ОГЭ",
]

CAT_FEATURES = [
    "student_target",
    "student_class",
    "course_name",
    "subject_name",
]

USED_FEATURES = [
    # Категориальные
    "student_target",
    "student_class",
    "course_name",
    "subject_name",
    # Базовые оценки
    "homework_done_mark",
    "test_part_one",
    "test_part_two",
    # Лаги
    "homework_lag_1",
    "homework_lag_2",
    "test1_lag_1",
    "test2_lag_1",
    # Разности
    "homework_diff",
    "test1_diff",
    "test2_diff",
    # Rolling-фичи
    "homework_rolling_mean_3",
    "homework_rolling_std_3",
    "test1_rolling_mean_3",
    "test2_rolling_std_3",
    # Экстремумы
    "homework_max",
    "homework_min",
    "test1_max",
    "test1_min",
    "test2_max",
    "test2_min",
]

TARGET_COL = "course_student_ege_result"
