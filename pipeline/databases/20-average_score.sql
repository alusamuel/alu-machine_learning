-- Stored procedure ComputeAverageScoreForUser to update average_score
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_s FLOAT;

    SELECT AVG(score) INTO avg_s
    FROM corrections
    WHERE corrections.user_id = user_id;

    UPDATE users
    SET average_score = IFNULL(avg_s, 0)
    WHERE id = user_id;
END$$

DELIMITER ;
