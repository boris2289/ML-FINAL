-- ══════════════════════════════════════════════════════════════
--  EGE Prediction ML System — схема БД
--  Весь жизненный цикл данных: raw → cleaned → prepared → predictions
-- ══════════════════════════════════════════════════════════════

-- ── 1. Сырые данные (загрузка CSV as-is) ─────────────────────
CREATE TABLE IF NOT EXISTS raw_data (
    id              BIGSERIAL PRIMARY KEY,
    student_id      DOUBLE PRECISION,
    student_target  TEXT,
    student_class   TEXT,
    course_type     TEXT,
    course_package_type TEXT,
    subject_name    TEXT,
    course_student_active DOUBLE PRECISION,
    course_student_ege_result DOUBLE PRECISION,
    homework_done_respectful TEXT,
    homework_done_mark DOUBLE PRECISION,
    test_part       DOUBLE PRECISION,
    test_done_mark  DOUBLE PRECISION,
    lesson_date     TIMESTAMP,
    student_city    TEXT,
    course_name     TEXT,
    homework_done_mark_probe TEXT,
    clan_name       TEXT,
    experiment      TEXT NOT NULL DEFAULT 'default',
    loaded_at       TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_experiment ON raw_data(experiment);

-- ── 2. Очищенные данные (после clean + lag/rolling) ──────────
CREATE TABLE IF NOT EXISTS cleaned_data (
    id              BIGSERIAL PRIMARY KEY,
    student_id      DOUBLE PRECISION,
    student_target  TEXT,
    student_class   TEXT,
    course_type     TEXT,
    course_package_type TEXT,
    subject_name    TEXT,
    course_student_active DOUBLE PRECISION,
    course_student_ege_result DOUBLE PRECISION,
    homework_done_respectful TEXT,
    homework_done_mark DOUBLE PRECISION,
    test_part_one   DOUBLE PRECISION,
    test_part_two   DOUBLE PRECISION,
    lesson_date     TIMESTAMP,
    student_city    TEXT,
    course_name     TEXT,
    homework_done_mark_probe TEXT,
    clan_name       TEXT,
    homework_lag_1  DOUBLE PRECISION,
    homework_lag_2  DOUBLE PRECISION,
    test1_lag_1     DOUBLE PRECISION,
    test1_lag_2     DOUBLE PRECISION,
    test2_lag_1     DOUBLE PRECISION,
    test2_lag_2     DOUBLE PRECISION,
    homework_diff   DOUBLE PRECISION,
    test1_diff      DOUBLE PRECISION,
    test2_diff      DOUBLE PRECISION,
    homework_rolling_mean_3 DOUBLE PRECISION,
    homework_rolling_std_3  DOUBLE PRECISION,
    test1_rolling_mean_3    DOUBLE PRECISION,
    test1_rolling_std_3     DOUBLE PRECISION,
    test2_rolling_mean_3    DOUBLE PRECISION,
    test2_rolling_std_3     DOUBLE PRECISION,
    homework_max    DOUBLE PRECISION,
    homework_min    DOUBLE PRECISION,
    test1_max       DOUBLE PRECISION,
    test1_min       DOUBLE PRECISION,
    test2_max       DOUBLE PRECISION,
    test2_min       DOUBLE PRECISION,
    experiment      TEXT NOT NULL DEFAULT 'default',
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cleaned_experiment ON cleaned_data(experiment);

-- ── 3. Подготовленные данные (агрегированные, без выбросов) ───
CREATE TABLE IF NOT EXISTS prepared_data (
    id              BIGSERIAL PRIMARY KEY,
    student_id      DOUBLE PRECISION,
    student_target  TEXT NOT NULL,
    student_class   TEXT NOT NULL,
    course_name     TEXT NOT NULL,
    subject_name    TEXT NOT NULL,
    course_type     TEXT,
    course_package_type TEXT,
    month           TEXT,
    course_student_ege_result DOUBLE PRECISION,
    homework_done_mark DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_one   DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_two   DOUBLE PRECISION NOT NULL DEFAULT 0,
    homework_lag_1  DOUBLE PRECISION DEFAULT -1,
    homework_lag_2  DOUBLE PRECISION DEFAULT -1,
    test1_lag_1     DOUBLE PRECISION DEFAULT -1,
    test2_lag_1     DOUBLE PRECISION DEFAULT -1,
    homework_diff   DOUBLE PRECISION DEFAULT 0,
    test1_diff      DOUBLE PRECISION DEFAULT 0,
    test2_diff      DOUBLE PRECISION DEFAULT 0,
    homework_rolling_mean_3 DOUBLE PRECISION DEFAULT -1,
    homework_rolling_std_3  DOUBLE PRECISION DEFAULT -1,
    test1_rolling_mean_3    DOUBLE PRECISION DEFAULT -1,
    test2_rolling_std_3     DOUBLE PRECISION DEFAULT -1,
    homework_max    DOUBLE PRECISION DEFAULT 0,
    homework_min    DOUBLE PRECISION DEFAULT 0,
    test1_max       DOUBLE PRECISION DEFAULT 0,
    test1_min       DOUBLE PRECISION DEFAULT 0,
    test2_max       DOUBLE PRECISION DEFAULT 0,
    test2_min       DOUBLE PRECISION DEFAULT 0,
    experiment      TEXT NOT NULL DEFAULT 'default',
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prepared_experiment ON prepared_data(experiment);
CREATE INDEX IF NOT EXISTS idx_prepared_subject ON prepared_data(subject_name);

-- ── 4. Входные данные для инференса ──────────────────────────
CREATE TABLE IF NOT EXISTS students_input (
    id              BIGSERIAL PRIMARY KEY,
    student_target  TEXT NOT NULL,
    student_class   TEXT NOT NULL,
    course_name     TEXT NOT NULL,
    subject_name    TEXT NOT NULL,
    homework_done_mark DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_one   DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_two   DOUBLE PRECISION NOT NULL DEFAULT 0,
    homework_lag_1  DOUBLE PRECISION DEFAULT -1,
    homework_lag_2  DOUBLE PRECISION DEFAULT -1,
    test1_lag_1     DOUBLE PRECISION DEFAULT -1,
    test2_lag_1     DOUBLE PRECISION DEFAULT -1,
    homework_diff   DOUBLE PRECISION DEFAULT 0,
    test1_diff      DOUBLE PRECISION DEFAULT 0,
    test2_diff      DOUBLE PRECISION DEFAULT 0,
    homework_rolling_mean_3 DOUBLE PRECISION DEFAULT -1,
    homework_rolling_std_3  DOUBLE PRECISION DEFAULT -1,
    test1_rolling_mean_3    DOUBLE PRECISION DEFAULT -1,
    test2_rolling_std_3     DOUBLE PRECISION DEFAULT -1,
    homework_max    DOUBLE PRECISION DEFAULT 0,
    homework_min    DOUBLE PRECISION DEFAULT 0,
    test1_max       DOUBLE PRECISION DEFAULT 0,
    test1_min       DOUBLE PRECISION DEFAULT 0,
    test2_max       DOUBLE PRECISION DEFAULT 0,
    test2_min       DOUBLE PRECISION DEFAULT 0,
    source_ege_result DOUBLE PRECISION,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ── 5. Предсказания ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                   BIGSERIAL PRIMARY KEY,
    input_data_id        BIGINT NOT NULL REFERENCES students_input(id) ON DELETE CASCADE,
    predicted_ege_score  DOUBLE PRECISION NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_version        TEXT,
    experiment           TEXT DEFAULT 'default',
    UNIQUE (input_data_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_input_id ON predictions(input_data_id);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_pred_experiment ON predictions(experiment);
