# Databases

This directory contains database exercises for data pipeline tasks. The tasks
cover SQL queries, MySQL table constraints, triggers, stored procedures,
MongoDB shell commands, PyMongo helpers, and basic MongoDB log statistics.

## SQL Files

- `0-create_database_if_missing.sql`: creates a database if it does not
  already exist.
- `1-first_table.sql`: creates a table with an ID and name.
- `2-list_values.sql`: lists rows from a table.
- `3-insert_value.sql`: inserts a row into a table.
- `4-best_score.sql`: lists records ordered by score.
- `5-average.sql`: calculates an average score.
- `6-avg_temperatures.sql`: calculates average temperatures by city.
- `7-max_state.sql`: lists maximum temperatures by state.
- `8-genre_id_by_show.sql`: joins shows to genre IDs.
- `9-no_genre.sql`: lists shows without a genre.
- `10-count_shows_by_genre.sql`: counts shows by genre.
- `11-rating_shows.sql`: ranks shows by total rating.
- `12-rating_genres.sql`: ranks genres by total rating.
- `13-uniq_users.sql`: creates a users table with unique email addresses.
- `14-country_users.sql`: creates a users table with country constraints.
- `15-fans.sql`: ranks origin countries by fan count.
- `16-glam_rock.sql`: lists Glam rock bands by lifespan.
- `17-store.sql`: creates a trigger to manage item quantities.
- `18-valid_email.sql`: creates a trigger for email validation.
- `19-bonus.sql`: creates a stored procedure for adding corrections.
- `20-average_score.sql`: creates a stored procedure for average scores.
- `21-div.sql`: creates a safe division function.

## MongoDB Files

- `22-list_databases`: lists MongoDB databases.
- `23-use_or_create_database`: switches to or creates a MongoDB database.
- `24-insert`: inserts a school document.
- `25-all`: lists school documents.
- `26-match`: lists schools matching a name.
- `27-count`: counts school documents.
- `28-update`: updates school documents.
- `29-delete`: deletes matching school documents.
- `30-all.py`: lists all documents in a PyMongo collection.
- `31-insert_school.py`: inserts a school document with PyMongo.
- `32-update_topics.py`: updates school topics by name.
- `33-schools_by_topic.py`: lists schools that contain a topic.
- `34-log_stats.py`: prints statistics for Nginx logs stored in MongoDB.
