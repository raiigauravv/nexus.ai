# Deployment Guide

Nexus-AI is designed as a full-stack ML platform, so the repository ships with a rich local development stack:

- FastAPI backend
- Next.js frontend
- PostgreSQL
- Redis
- MinIO
- Kafka + Zookeeper
- MLflow

That setup is appropriate for local development and demos, but it does not fit cleanly into AWS free tier without trimming the architecture.

## What AWS Free Tier Can Realistically Handle

AWS free tier can support a **reduced** version of the project if you keep the deployment intentionally small:

- Frontend on Vercel or AWS Amplify
- Backend on a single small EC2 instance
- One managed database, if you accept the free-tier limits and one-year clock
- No Kafka, no MinIO, no Redis cluster, no MLflow server in the deployed path

That means the deployed app should be treated as a product shell with core inference endpoints, not the full internal MLOps stack.

## Recommended Low-Cost Production Shape

### Option 1: Cheapest practical deployment

- Frontend: Vercel
- Backend: single EC2 instance
- Database: Amazon RDS free-tier PostgreSQL, or SQLite for a demo-only build
- Storage: local disk or S3 for only the essentials
- Async/eventing: disabled or replaced with simple in-process jobs

This is the easiest way to keep the public app responsive while avoiding a multi-container AWS bill.

### Option 2: Full stack, but not free tier

- Frontend: Vercel or Amplify
- Backend: ECS/Fargate or EC2
- PostgreSQL: RDS
- Redis: ElastiCache or local instance
- Kafka: MSK or a queue alternative
- MinIO: S3
- MLflow: ECS, EC2, or managed artifact storage

This is the right shape if you want the full platform experience preserved in deployment.

## What To Remove For Free Tier

If you want an AWS free-tier-friendly build, remove or disable these services from the deployed path:

- Kafka/Zookeeper
- MinIO
- MLflow server
- Redis dependency unless you truly need caching
- Background consumers that expect a durable event stream

The current codebase already works as a local showcase for these systems, but the deployed version should be simpler.

## Honest Recommendation

If the goal is portfolio polish, the best move is:

1. Keep the full stack for local development and demos.
2. Publish the frontend separately.
3. Deploy a trimmed backend with one database.
4. Document which ML and infra features are demo-only.

That gives you a deployment story that is both credible and affordable.

## Operational Notes

- The repository root `docker-compose.yml` is still the best way to run the complete system locally.
- Free-tier deployment should be treated as a separate “lite” target, not a replacement for the full development stack.
- If you want, the next step is to add a dedicated `docker-compose.lite.yml` and env profile for the reduced deployment.