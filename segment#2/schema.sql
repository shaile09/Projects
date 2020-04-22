
 -- Creating tables for BUSINESS_REVIEWSDB
CREATE TABLE businesses (
     business_id VARCHAR(50) NOT NULL,
     name VARCHAR(200) NOT NULL,
     PRIMARY KEY (business_id),
     --UNIQUE (business_id)
);
CREATE TABLE business_info (
	business_id VARCHAR(50) NOT NULL,
    city VARCHAR NOT NULL,
    state VARCHAR (2) NOT NULL,
    postal_code INT NOT NULL,
    latitude INT NOT NULL,
    longitude INT NOT NULL,
    review_count INT NOT NULL,
    EthnicType VARCHAR NOT NULL,
    stars VARCHAR NOT NULL,
--FOREIGN KEY (business_id) REFERENCES businesses (business_id),
    PRIMARY KEY (business_id)
);
CREATE TABLE business_reviews (
    review_id VARCHAR (50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    business_id VARCHAR (50) NOT NULL,
    date DATE NOT NULL,
    review_star INT NOT NULL,
    text VARCHAR NOT NULL,
    useful INT NOT NULL,
    city VARCHAR NOT NULL,
    state VARCHAR (2) NOT NULL,
    postal_code INT NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    EthnicType VARCHAR NOT NULL,
--FOREIGN KEY (business_id) REFERENCES businesses (business_id),
--FOREIGN KEY (business_id) REFERENCES bussiness_info (business_id),
	PRIMARY KEY (review_id)
);
CREATE TABLE mlbusiness_reviews (
    review_id VARCHAR (50) NOT NULL,
    business_id VARCHAR (50) NOT NULL,
    review_star INT NOT NULL,
    useful INT NOT NULL,
    EthnicType VARCHAR NOT NULL,
    city VARCHAR NOT NULL,
    state VARCHAR (2) NOT NULL,
    postal_code INT NOT NULL,
--FOREIGN KEY (business_id) REFERENCES businesses (business_id),
--FOREIGN KEY (review_id) REFERENCES buss_reviews (review_id),
	PRIMARY KEY (review_id)
);

SELECT * FROM businesses;
SELECT * FROM business_info;
SELECT * FROM business_reviews;
SELECT * FROM mlbusiness_reviews;
