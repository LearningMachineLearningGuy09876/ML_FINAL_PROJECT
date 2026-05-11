import os
import pandas as pd


class UserProfileStore:
    """
    Stores and manages user movie ratings for the StreamSense / FlikPik app.
    """

    def __init__(self, ratings_path):
        self.ratings_path = ratings_path

        if not os.path.exists(self.ratings_path):

            os.makedirs(
                os.path.dirname(self.ratings_path),
                exist_ok=True
            )

            empty_df = pd.DataFrame(
                columns=[
                    "username",
                    "movie_id",
                    "title",
                    "rating"
                ]
            )

            empty_df.to_csv(self.ratings_path, index=False)

        self.df = pd.read_csv(self.ratings_path)

    def get_users(self):
        if self.df.empty:
            return []

        return sorted(
            self.df["username"]
            .dropna()
            .unique()
            .tolist()
        )

    def get_user_ratings(self, username):

        if self.df.empty:
            return pd.DataFrame()

        return self.df[
            self.df["username"] == username
        ]

    def add_or_update_rating(
        self,
        username,
        movie_id,
        title,
        rating,
    ):

        existing_mask = (
            (self.df["username"] == username)
            &
            (self.df["movie_id"] == movie_id)
        )

        if existing_mask.any():

            self.df.loc[
                existing_mask,
                "rating"
            ] = rating

        else:

            new_row = pd.DataFrame([
                {
                    "username": username,
                    "movie_id": movie_id,
                    "title": title,
                    "rating": rating,
                }
            ])

            self.df = pd.concat(
                [self.df, new_row],
                ignore_index=True,
            )

        self.df.to_csv(
            self.ratings_path,
            index=False,
        )

    def delete_user(self, username):

        self.df = self.df[
            self.df["username"] != username
        ]

        self.df.to_csv(
            self.ratings_path,
            index=False,
        )

    def delete_rating(
        self,
        username,
        movie_id,
    ):

        self.df = self.df[
            ~(
                (self.df["username"] == username)
                &
                (self.df["movie_id"] == movie_id)
            )
        ]

        self.df.to_csv(
            self.ratings_path,
            index=False,
        )

    def get_user_stats(self, username):

        ratings = self.get_user_ratings(username)

        if ratings.empty:
            return {
                "movies_rated": 0,
                "average_rating": 0,
                "highest_rating": 0,
            }

        return {
            "movies_rated": len(ratings),
            "average_rating": round(
                ratings["rating"].mean(),
                2,
            ),
            "highest_rating": round(
                ratings["rating"].max(),
                2,
            ),
        }
