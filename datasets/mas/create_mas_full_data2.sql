create table if not exists mas.mas_full_data2
(
    _id                            int auto_increment primary key,
    author$aid                     int           not null,
    author$name                    varchar(60)   null,
    author$homepage                varchar(200)  null,
    author$photo                   varchar(200)  null,
    author$paper_count             int           null,
    author$citation_count          int           null,
    author$importance              int           null,
    publication$pid                int           not null,
    publication$title              varchar(200)  null,
    publication$abstract           varchar(2000) null,
    publication$year               int           null,
    publication$doi                varchar(200)  null,
    publication$citation_count     int           null,
    publication$reference_count    int           null,
    publication$importance         int           null,
    publication$conference_journal varchar(100)  null,
    organization$oid               int           not null,
    organization$name              varchar(150)  null,
    organization$name_short        varchar(45)   null,
    organization$continent         varchar(45)   null,
    organization$homepage          varchar(200)  null,
    organization$author_count      int           null,
    organization$paper_count       int           null,
    organization$citation_count    int           null,
    organization$importance        int           null,
    conference$cid                 int           not null,
    conference$name                varchar(100)  null,
    conference$full_name           varchar(200)  null,
    conference$homepage            varchar(200)  null,
    conference$paper_count         int           null,
    conference$citation_count      int           null,
    conference$importance          int           null,
    domain$did                     int           not null,
    domain$name                    varchar(45)   null,
    domain$paper_count             int           null,
    domain$importance              int           null,
    keyword$importance             int           null,
    keyword$keyword                varchar(100)  null,
    keyword$keyword_short          varchar(45)   null,
    keyword$kid                    int           not null
);

CREATE INDEX author$paper_count_idx ON mas.mas_full_data2 (author$paper_count);
CREATE INDEX publication$year_idx ON mas.mas_full_data2 (publication$year);
CREATE INDEX keyword$keyword_idx ON mas.mas_full_data2 (keyword$keyword);
CREATE INDEX organization$name_idx ON mas.mas_full_data2 (organization$name);
CREATE INDEX publication$citation_count_idx ON mas.mas_full_data2 (publication$citation_count);
CREATE INDEX domain$name_idx ON mas.mas_full_data2 (domain$name);
CREATE INDEX author$citation_count_idx ON mas.mas_full_data2 (author$citation_count);
CREATE INDEX conference$name_idx ON mas.mas_full_data2 (conference$name);
CREATE INDEX author$name_idx ON mas.mas_full_data2 (author$name);
CREATE INDEX publication$title_idx ON mas.mas_full_data2 (publication$title);