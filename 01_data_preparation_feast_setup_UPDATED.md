# PostgreSQL Data Loading - Add to Notebook 01

Insert this cell **after** the current cell that saves to Parquet (Cell 18):

```python
# ===================================================================
# LOAD TO POSTGRESQL (Production Architecture)
# ===================================================================
print('\n📊 Loading features to PostgreSQL...')

import sqlalchemy as sa
from sqlalchemy import create_engine

# PostgreSQL connection (adjust for your environment)
# For local: localhost:5432
# For OpenShift: feast-postgres.kft-feast-quickstart.svc.cluster.local:5432
PG_HOST = os.getenv('FEAST_PG_HOST', 'feast-postgres.kft-feast-quickstart.svc.cluster.local')
PG_PORT = os.getenv('FEAST_PG_PORT', '5432')
PG_USER = os.getenv('FEAST_PG_USER', 'feast')
PG_PASSWORD = os.getenv('FEAST_PG_PASSWORD', 'feast_password')
PG_DATABASE = 'feast_offline'

# Connection string using psycopg (PostgreSQL driver)
connection_string = f'postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}'

try:
    engine = create_engine(connection_string, pool_pre_ping=True)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(sa.text("SELECT version()"))
        pg_version = result.fetchone()[0]
        print(f'✓ Connected to PostgreSQL: {pg_version.split(",")[0]}')
    
    # Load sales_features to PostgreSQL
    print(f'\n  Loading sales_features table ({len(sales_df):,} rows)...')
    start_time = time.time()
    sales_df.to_sql(
        'sales_features',
        engine,
        if_exists='replace',  # Replace table if exists
        index=False,
        method='multi',  # Batch inserts for performance
        chunksize=10000
    )
    sales_elapsed = time.time() - start_time
    print(f'  ✓ Loaded in {sales_elapsed:.1f}s ({len(sales_df)/sales_elapsed:.0f} rows/sec)')
    
    # Load store_features to PostgreSQL
    print(f'\n  Loading store_features table ({len(store_expanded_df):,} rows)...')
    start_time = time.time()
    store_expanded_df.to_sql(
        'store_features',
        engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=10000
    )
    store_elapsed = time.time() - start_time
    print(f'  ✓ Loaded in {store_elapsed:.1f}s ({len(store_expanded_df)/store_elapsed:.0f} rows/sec)')
    
    # Create indexes for fast retrieval
    print(f'\n  Creating indexes...')
    with engine.connect() as conn:
        # Sales features indexes
        conn.execute(sa.text('CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_features(date)'))
        conn.execute(sa.text('CREATE INDEX IF NOT EXISTS idx_sales_store_dept ON sales_features(store, dept)'))
        conn.execute(sa.text('CREATE INDEX IF NOT EXISTS idx_sales_store_dept_date ON sales_features(store, dept, date)'))
        
        # Store features indexes
        conn.execute(sa.text('CREATE INDEX IF NOT EXISTS idx_store_date ON store_features(date)'))
        conn.execute(sa.text('CREATE INDEX IF NOT EXISTS idx_store_store_dept ON store_features(store, dept)'))
        conn.execute(sa.text('CREATE INDEX IF NOT EXISTS idx_store_store_dept_date ON store_features(store, dept, date)'))
        
        conn.commit()
    print(f'  ✓ Created indexes for fast queries')
    
    # Verify data loaded correctly
    print(f'\n  Verifying data...')
    with engine.connect() as conn:
        sales_count = conn.execute(sa.text('SELECT COUNT(*) FROM sales_features')).fetchone()[0]
        store_count = conn.execute(sa.text('SELECT COUNT(*) FROM store_features')).fetchone()[0]
        
        print(f'  ✓ sales_features: {sales_count:,} rows')
        print(f'  ✓ store_features: {store_count:,} rows')
        
        # Show sample data
        sample_sales = conn.execute(sa.text('SELECT * FROM sales_features LIMIT 2')).fetchall()
        print(f'\n  Sample from sales_features:')
        print(f'    First row: store={sample_sales[0][0]}, dept={sample_sales[0][1]}, weekly_sales={sample_sales[0][3]:.2f}')
    
    print(f'\n✅ PostgreSQL load complete!')
    print(f'   Connection: {PG_HOST}:{PG_PORT}/{PG_DATABASE}')
    print(f'   Tables: sales_features, store_features')
    print(f'   Indexes: 6 indexes created for fast retrieval')
    
except Exception as e:
    print(f'\n⚠️  PostgreSQL load failed: {e}')
    print(f'   Reason: {type(e).__name__}')
    print(f'   Falling back to Parquet files only (still usable for training)')
    print(f'\n   To fix: Ensure PostgreSQL is running and accessible at {PG_HOST}:{PG_PORT}')
```

## Update Notebook Cell 2 (Install Dependencies)

Replace the existing cell:

```python
%pip install feast==0.56.0 kaggle==1.7.4.5 pandas==2.2.3 pyarrow==17.0.0 scikit-learn==1.6.1 psycopg==3.1.18 sqlalchemy==2.0.36 ray[default]==2.35.0
```

Added:
- `feast==0.56.0` (upgraded from 0.54.0 for PostgreSQL + Ray support)
- `psycopg==3.1.18` (PostgreSQL driver, replaces psycopg2-binary)
- `sqlalchemy==2.0.36` (database toolkit)
- `ray[default]==2.35.0` (distributed computing)

## Update Notebook Cell 20 (Feast Apply)

Before running `feast apply`, set environment variables for the PostgreSQL connection:

```python
# Set PostgreSQL environment variables for Feast
os.environ['FEAST_PG_HOST'] = 'feast-postgres.kft-feast-quickstart.svc.cluster.local'
os.environ['FEAST_PG_PORT'] = '5432'
os.environ['FEAST_PG_USER'] = 'feast'
os.environ['FEAST_PG_PASSWORD'] = 'feast_password'

# Apply feature definitions to Feast
!cd /shared/feature_repo && feast apply
```



