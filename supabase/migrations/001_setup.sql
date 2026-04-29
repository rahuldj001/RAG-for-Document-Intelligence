-- Run this SQL once in the Supabase SQL editor before ingesting documents.

create extension if not exists vector;

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(384) not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists documents_embedding_ivfflat_idx
on public.documents
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create index if not exists documents_metadata_gin_idx
on public.documents
using gin (metadata);

create or replace function public.match_documents(
  query_embedding vector(384),
  match_count int,
  filename_filter text default null
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from public.documents
  where filename_filter is null
     or documents.metadata ->> 'filename' = filename_filter
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;

create table if not exists public.eval_results (
  id uuid primary key default gen_random_uuid(),
  run_name text not null,
  scores jsonb not null,
  deltas jsonb not null default '{}'::jsonb,
  sample_count int not null default 0,
  created_at timestamptz not null default now()
);

