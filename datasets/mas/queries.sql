SELECT _id FROM mas.mas_full_data2 WHERE publication$title LIKE 'Making %' AND conference$name = 'VLDB'; # Added pid=1507687
SELECT _id FROM mas.mas_full_data2 WHERE domain$name = 'Databases';
SELECT _id FROM mas.mas_full_data2 WHERE domain$name = 'Machine Learning & Pattern Recognition';
SELECT _id FROM mas.mas_full_data2 WHERE keyword$keyword = 'Machine Learning';
SELECT _id FROM mas.mas_full_data2 WHERE keyword$keyword = 'Database Query';
SELECT _id FROM mas.mas_full_data2 WHERE keyword$keyword = 'Natural Language Processing';
SELECT _id FROM mas.mas_full_data2 WHERE author$paper_count > 10 AND organization$name = 'Tel Aviv University' AND publication$year > 2010 AND publication$citation_count > 1; # Added pid IN (432646, 2666921, 1821625)
SELECT _id FROM mas.mas_full_data2 WHERE author$paper_count < 100 AND author$citation_count > 1000 AND organization$name = 'University of California San Diego' AND publication$year > 2010; # Added pid IN (1530388, 2679889)
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'University of Michigan' AND publication$year > 2010;
SELECT _id FROM mas.mas_full_data2 WHERE conference$name = 'VLDB' AND ( publication$year < 1995 or publication$year > 2002 ) AND publication$citation_count > 100;
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'Tel Aviv University' AND publication$year > 2010;
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'University of California San Diego' AND publication$year > 2010;
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'University of Michigan' AND publication$year > 2010;
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'Tel Aviv University' AND publication$year > 2010 AND domain$name = 'Databases';
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'University of California San Diego' AND publication$year > 2010 AND domain$name = 'Databases';
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'University of Michigan' AND publication$year > 2010 AND domain$name = 'Databases';
SELECT _id FROM mas.mas_full_data2 WHERE organization$name = 'University of Michigan' AND publication$year > 2010 AND domain$name = 'Databases' AND author$paper_count > 20;
SELECT _id FROM mas.mas_full_data2 WHERE author$name = 'Tova Milo';
SELECT _id FROM mas.mas_full_data2 WHERE author$name = 'H. V. Jagadish';
SELECT _id FROM mas.mas_full_data2 WHERE author$name = 'Alin Deutsch';
SELECT _id FROM mas.mas_full_data2 WHERE publication$year > 2010 AND ( author$name = 'Tova Milo' or author$name = 'H. V. Jagadish' );
SELECT _id FROM mas.mas_full_data2 WHERE publication$year > 2010 AND ( author$name = 'Tova Milo' or author$name = 'H. V. Jagadish' );
SELECT _id FROM mas.mas_full_data2 WHERE conference$name = 'SIGMOD' AND organization$name = 'University of California San Diego' AND publication$year = 2005;
SELECT _id FROM mas.mas_full_data2 WHERE conference$name = 'SIGMOD' AND organization$name = 'University of Michigan' AND publication$year = 2005;