CREATE TABLE joined_titles1 AS
SELECT new_title.id                        AS `title$id`,
       new_title.title                     AS `title$title`,
       new_title.imdb_index                AS `title$imdb_index`,
       new_title.kind_id                   AS `title$kind_id`,
       new_title.production_year           AS `title$production_year`,
       new_title.imdb_id                   AS `title$imdb_id`,
       new_title.phonetic_code             AS `title$phonetic_code`,
       new_title.episode_of_id             AS `title$episode_of_id`,
       new_title.season_nr                 AS `title$season_nr`,
       new_title.episode_nr                AS `title$episode_nr`,
       new_title.series_years              AS `title$series_years`,
       new_title.md5sum                    AS `title$md5sum`,
       new_movie_companies.id              AS `movie_companies$id`,
       new_movie_companies.movie_id        AS `movie_companies$movie_id`,
       new_movie_companies.company_id      AS `movie_companies$company_id`,
       new_movie_companies.company_type_id AS `movie_companies$company_type_id`,
       new_movie_companies.note            AS `movie_companies$note`,
       movie_keyword.id                    AS `movie_keyword$id`,
       movie_keyword.movie_id              AS `movie_keyword$movie_id`,
       movie_keyword.keyword_id            AS `movie_keyword$keyword_id`
FROM new_title,
     new_movie_companies,
     movie_keyword
WHERE new_title.id = new_movie_companies.movie_id
  AND new_title.id = movie_keyword.movie_id;

ALTER TABLE joined_titles1 ADD INDEX (movie_companies$company_type_id);
ALTER TABLE joined_titles1 ADD INDEX (title$production_year);
ALTER TABLE joined_titles1 ADD INDEX (movie_keyword$keyword_id);
ALTER TABLE joined_titles1 ADD INDEX (title$kind_id);
ALTER TABLE joined_titles1 ADD INDEX (movie_companies$company_id);

CREATE TABLE join_title_companies_keyword
(
    _id                             int          NOT NULL AUTO_INCREMENT,
    title$id                        int(11)      NOT NULL,
    title$title                     varchar(500) NOT NULL,
    title$imdb_index                varchar(5)   DEFAULT NULL,
    title$kind_id                   int(11)      NOT NULL,
    title$production_year           int(11)      DEFAULT NULL,
    title$imdb_id                   int(11)      DEFAULT NULL,
    title$phonetic_code             varchar(5)   DEFAULT NULL,
    title$episode_of_id             int(11)      DEFAULT NULL,
    title$season_nr                 int(11)      DEFAULT NULL,
    title$episode_nr                int(11)      DEFAULT NULL,
    title$series_years              varchar(49)  DEFAULT NULL,
    title$md5sum                    varchar(32)  DEFAULT NULL,
    movie_companies$id              int(11)      NOT NULL,
    movie_companies$movie_id        int(11)      NOT NULL,
    movie_companies$company_id      int(11)      NOT NULL,
    movie_companies$company_type_id int(11)      NOT NULL,
    movie_companies$note            varchar(250) DEFAULT NULL,
    movie_keyword$id                int(11)      NOT NULL,
    movie_keyword$movie_id          int(11)      NOT NULL,
    movie_keyword$keyword_id        int(11)      NOT NULL,
    PRIMARY KEY (_id),
    KEY title$id_idx (title$id),
    KEY movie_companies$company_type_id_idx (movie_companies$company_type_id),
    KEY title$production_year_idx (title$production_year),
    KEY movie_keyword$keyword_id_idx (movie_keyword$keyword_id),
    KEY title$kind_id_idx (title$kind_id),
    KEY movie_companies$company_id_idx (movie_companies$company_id)
);
ALTER TABLE join_title_companies_keyword
    AUTO_INCREMENT = 0;