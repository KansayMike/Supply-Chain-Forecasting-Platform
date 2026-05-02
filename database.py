# database.py
# This file's job: Give other Python files an easy way to connect to our database

from sqlalchemy import create_engine
# SQLAlchemy is a Python library that lets you write Python code
# instead of raw SQL. It's like a universal remote for databases.

from sqlalchemy.orm import sessionmaker
# sessionmaker creates "sessions" — think of them as temporary connections
# where you do work and then commit (save) or rollback (undo).

from sqlalchemy import create_engine, text

import os
# os lets us read environment variables (settings from the computer)

def get_engine():
    """
    This function creates and returns a "database engine."
    An engine is like a phone line to the database — it knows how to connect,
    but it doesn't actually connect until you make a call.
    """
    
    # We're building the connection string piece by piece
    # Format: postgresql://username:password@host:port/database_name
    
    user = "admin"           # Matches POSTGRES_USER in docker-compose.yml
    password = "secret"      # Matches POSTGRES_PASSWORD
    host = "db"              # Matches the service name in docker-compose.yml
                             # Docker automatically makes this a network address
    port = "5432"            # The default PostgreSQL port
    database = "supply_chain"  # Matches POSTGRES_DB
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    # The 'f' before the string means "format string" — variables inside {} get inserted
    
    engine = create_engine(connection_string)
    # create_engine doesn't connect immediately. It just prepares the connection.
    
    return engine


def get_session():
    """
    This function creates a session — an active conversation with the database.
    Use it like:
        session = get_session()
        # ... do database work ...
        session.commit()  # Save changes
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    # "Make a session factory bound to this engine"
    
    return Session()


def test_connection():
    """
    This function tests if we can actually reach the database.
    Run this to verify everything is set up correctly.
    """
    try:
        engine = get_engine()
        # Connect and run a simple query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))

            # "SELECT version();" asks PostgreSQL what version it is
            
            version = result.fetchone()
            # fetchone() gets the first row of results
            
            print(f"✅ Database connected successfully!")
            print(f"   PostgreSQL version: {version[0]}")
            return True
            
    except Exception as e:
        # If anything goes wrong, we end up here
        print(f"❌ Failed to connect to database:")
        print(f"   Error: {e}")
        return False


# This block only runs if you execute this file directly
# (not when you import it from another file)
if __name__ == "__main__":
    test_connection()