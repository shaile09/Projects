-- Creating tables for BUSINESS_REVIEWSDB
CREATE TABLE businesses (
     bussiness_id VARCHAR(50) NOT NULL,
     name VARCHAR(50) NOT NULL,
     PRIMARY KEY (buss_id),
     UNIQUE (buss_id)
);
CREATE TABLE bussiness_info (
	bussiness_id VARCHAR(50) NOT NULL,
    name VARCHAR(50) NOT NULL,
    review_count INT NOT NULL,
    categories VARCHAR NOT NULL,
    stars VARCHAR NOT NULL,
    latitude INT NOT NULL,
    longtitude INT NOT NULL,
    postal_code INT NOT NULL,
    state VARCHAR (2) NOT NULL,
FOREIGN KEY (bussiness_id) REFERENCES businesses (bussiness_id),
    PRIMARY KEY (bussiness_id)
);
CREATE TABLE reviews (
    review_id VARCHAR (50) NOT NULL,
    bussiness_id VARCHAR (50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
	date DATE NOT NULL,
	stars INT NOT NULL,
FOREIGN KEY (bussiness_id) REFERENCES businesses (bussiness_id),
FOREIGN KEY (bussiness_id) REFERENCES bussiness_info (bussiness_id),
	PRIMARY KEY (user_id)
);

SELECT * FROM businesses;
SELECT * FROM buss_info;
SELECT * FROM reviews;