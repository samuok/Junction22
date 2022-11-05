DROP DATABASE IF EXISTS junction;
CREATE DATABASE junction;
USE junction;

CREATE TABLE junction (
	id INT NOT NULL AUTO_INCREMENT,
	arvo1 INT,
	arvo2 INT,
	arvo3 INT,
	arvo4 INT,
	arvo5 INT,
	arvo6 INT,
	arvo7 INT,
	arvo8 INT,
	terapeutti INT,
	
	PRIMARY KEY (id)
);

INSERT INTO junction VALUES (1, 1, 2, 2, 5, 4, 1, 5, 6, 1);
INSERT INTO junction VALUES (2, 3, 2, 2, 8, 9, 5, 6, 8, 2);

DROP USER IF EXISTS kayttis@localhost;
CREATE USER kayttis@localhost IDENTIFIED BY 'salis';
GRANT SELECT, INSERT, UPDATE, DELETE, DROP ON junction.* TO kayttis@localhost;


 
