CREATE TABLE IF NOT EXISTS students_input (
    id BIGSERIAL PRIMARY KEY,
    student_target TEXT NOT NULL,
    student_class TEXT NOT NULL,
    course_name TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    homework_done_mark DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_one DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_two DOUBLE PRECISION NOT NULL DEFAULT 0,
    homework_lag_1 DOUBLE PRECISION DEFAULT -1,
    homework_lag_2 DOUBLE PRECISION DEFAULT -1,
    test1_lag_1 DOUBLE PRECISION DEFAULT -1,
    test2_lag_1 DOUBLE PRECISION DEFAULT -1,
    homework_diff DOUBLE PRECISION DEFAULT 0,
    test1_diff DOUBLE PRECISION DEFAULT 0,
    test2_diff DOUBLE PRECISION DEFAULT 0,
    homework_rolling_mean_3 DOUBLE PRECISION DEFAULT -1,
    homework_rolling_std_3 DOUBLE PRECISION DEFAULT -1,
    test1_rolling_mean_3 DOUBLE PRECISION DEFAULT -1,
    test2_rolling_std_3 DOUBLE PRECISION DEFAULT -1,
    homework_max DOUBLE PRECISION DEFAULT 0,
    homework_min DOUBLE PRECISION DEFAULT 0,
    test1_max DOUBLE PRECISION DEFAULT 0,
    test1_min DOUBLE PRECISION DEFAULT 0,
    test2_max DOUBLE PRECISION DEFAULT 0,
    test2_min DOUBLE PRECISION DEFAULT 0,
    source_ege_result DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    input_data_id BIGINT NOT NULL REFERENCES students_input(id) ON DELETE CASCADE,
    predicted_ege_score DOUBLE PRECISION NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_version TEXT,
    UNIQUE (input_data_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_input_id ON predictions(input_data_id);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(prediction_timestamp);
