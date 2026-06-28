-- Function SafeDiv that divides a by b or returns 0 if b = 0
DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE res FLOAT;

    IF b = 0 THEN
        SET res = 0;
    ELSE
        SET res = a / b;
    END IF;

    RETURN res;
END$$

DELIMITER ;
