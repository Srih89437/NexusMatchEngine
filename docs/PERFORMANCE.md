# Performance Optimization Metrics

- **Pipeline Latency**: Average LTR scoring features extraction evaluates under `0.0072s`.
- **Concurrency**: Relational pools pre-ping connections and celery distributed workers handle high parsing workloads.
- **Payload indexes**: Creates Keyword indexes for `skills` and Float indexes for `years_exp` payload keys on database collection setup.
