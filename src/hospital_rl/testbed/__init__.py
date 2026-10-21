"""Optional isolated synthetic-service testbed (never a prerequisite for the graph benchmark).

Invented records only (``SYN-`` ids, no names/dates of birth/addresses), bounded data, deterministic seed + reset,
allowlisted typed actions only (the same action vocabulary as the graph env: no SQL, no shell), audit storage the actor
API cannot reach, and an internal-only network policy that denies egress and provider credentials. The local
implementation uses stdlib sqlite as a stand-in for the PostgreSQL service (``SCHEMA`` is plain portable SQL);
a real PostgreSQL deployment is NOT built or run here. Nothing here claims real-hospital validity or HIPAA compliance.
"""
