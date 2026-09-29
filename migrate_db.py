from sqlalchemy import inspect, text
from app import app, db


def add_column_if_missing(
    table_name,
    column_name,
    column_definition
):
    inspector = inspect(db.engine)

    columns = [
        column["name"]
        for column in inspector.get_columns(
            table_name
        )
    ]

    if column_name not in columns:

        print(
            f"Adding column: "
            f"{table_name}.{column_name}"
        )

        db.session.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN {column_name} "
                f"{column_definition}"
            )
        )

        db.session.commit()

    else:

        print(
            f"Already exists: "
            f"{table_name}.{column_name}"
        )


with app.app_context():

    print("Checking database...")

    add_column_if_missing(
        "report",
        "probabilities",
        "TEXT"
    )

    add_column_if_missing(
        "report",
        "gradcam",
        "VARCHAR(255)"
    )

    add_column_if_missing(
        "report",
        "review_status",
        "VARCHAR(30) DEFAULT 'Pending'"
    )

    add_column_if_missing(
        "report",
        "doctor_comment",
        "TEXT"
    )

    add_column_if_missing(
        "report",
        "ai_explanation",
        "TEXT"
    )

    print()
    print("Database migration completed!")