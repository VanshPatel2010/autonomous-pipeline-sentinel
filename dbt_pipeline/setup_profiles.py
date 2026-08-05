import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")
url = urlparse(db_url)
port = 6543 if "pooler.supabase.com" in url.hostname else url.port

profile_yml = f"""
dbt_pipeline:
  target: dev
  outputs:
    dev:
      type: postgres
      host: {url.hostname}
      user: {url.username}
      password: {url.password}
      port: {port}
      dbname: {url.path[1:]}
      schema: public
      threads: 1
      keepalives_idle: 0
"""
with open("dbt_pipeline/profiles.yml", "w") as f:
    f.write(profile_yml)
