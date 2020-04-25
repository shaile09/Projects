-- Creating tables for Yelp_DB
CREATE TABLE business_info (
	business_id VARCHAR(50) NOT NULL,
    business_name VARCHAR(200) NOT NULL,
    city VARCHAR NOT NULL,
    us_state VARCHAR (2) NOT NULL,
    postal_code INT NOT NULL,
    latitude INT NOT NULL,
    longitude INT NOT NULL,
    review_count INT NOT NULL,
    ethnic_type VARCHAR NOT NULL,
    stars VARCHAR NOT NULL,
    PRIMARY KEY (business_id)
);

CREATE TABLE business_reviews (
    review_id VARCHAR (50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    business_id VARCHAR (50) NOT NULL,
    review_date DATE NOT NULL,
    review_star INT NOT NULL,
    review VARCHAR NOT NULL,
    useful INT NOT NULL,
	PRIMARY KEY (review_id)
);

CREATE TABLE review_prediction (
    id INT,
    postal_code text,
    city text,
    ethnic_type text,
    prediction double precision
);

SELECT * FROM business_info;
SELECT * FROM business_reviews;
SELECT * FROM review_prediction;