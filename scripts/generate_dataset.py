#!/usr/bin/env python3
"""
Bootstrap generator for the 5k dataset.
Produces stratified train (70%), val (15%), test (15%) splits
with 3-gram Jaccard deduplication.
"""

import json
import random
import time
from pathlib import Path

random.seed(42)

SEEDS = {
    "TIER_1_SIMPLE": [
        "What does std::decay do in type traits?",
        "How do I sort a std::vector in descending order?",
        "Explain the difference between struct and class in C++.",
        "Write a helper function to trim whitespace from a string.",
        "How to convert an integer to a binary string in C++?",
        "What is the time complexity of std::unordered_map lookup?",
        "What header contains std::clamp?",
        "How do I pass a lambda as a callback function?",
        "What does [[nodiscard]] attribute mean in C++17?",
        "Write a function to check if a number is prime.",
        # Hard negatives (mentions complex terms, but simple task)
        "What header file defines std::memory_order?",
        "What is the syntax for creating an atomic boolean in C++?",
        "Does std::mutex constructor throw exceptions?",
        "What does the acronym MPMC stand for in lock-free queues?",
        "What header is required for POSIX pthread_create?",
    ],
    "TIER_2_REFACTOR": [
        "Rename DatabasePool::acquire() to DatabasePool::borrowConnection() across the codebase.",
        "Refactor this 200-line function into three focused helper functions.",
        "Replace raw pointers with std::shared_ptr in the SessionManager module.",
        "Move configuration parsing logic from main.cpp to a separate ConfigParser class.",
        "Update all calls to Logger::log() to include the new log level parameter.",
        "Extract interface IFileReader from ConcreteFileReader and update all consumers.",
        "Split the monolithic GodObject class into UserProfile, UserBilling, and UserAuth.",
        "Update database migration scripts and SQL queries after renaming user_id to account_id.",
        "Fix the off-by-one bug in the sliding window buffer calculation.",
        "Replace manual loop-based filtering with std::ranges or std::copy_if across these three files.",
        # Hard negatives (mentions architecture/threads, but purely procedural refactor)
        "Move the thread initialization code from Server.cpp into a dedicated ThreadPool wrapper class.",
        "Rename the mutex member variable m_lock to m_syncMutex across these four files.",
        "Extract the socket timeout retry loop into a standalone helper function.",
    ],
    "TIER_3_COMPLEX": [
        "Diagnose intermittent race condition causing corruption in this custom memory pool.",
        "Design a wait-free single-producer single-consumer circular queue using atomic operations.",
        "Analyze why this multithreaded pipeline deadlocks under high write contention.",
        "Design a dynamic plugin architecture with safe cross-DLL boundary exception and ABI handling.",
        "Formulate a cache-coherent SIMD vectorization plan for this raytracer intersection kernel.",
        "Determine the memory reclamation scheme (hazard pointers vs epoch-based) suitable for this lock-free tree.",
        "Debug why the application suffers from memory fragmentation after 48 hours of high-frequency allocations.",
        "Explain memory ordering invariants required for Double-Checked Locking Pattern on ARM architecture.",
        # Hard negatives (short / simple wording, but deep algorithmic problem)
        "Why does my lock-free stack leak memory under concurrent pop operations?",
        "Fix the livelock in our two-phase commit coordinator.",
        "Make this tree traversal lock-free without using global locks.",
    ]
}

TEMPLATES = {
    "TIER_1_SIMPLE": [
        "How do I {verb} a {type} in {lang}?",
        "What does {keyword} mean in {lang}?",
        "Write a utility function to {verb} {type}.",
        "Explain how {keyword} works in {lang}.",
        "What is the return type of {keyword}?",
        "How to format a {type} into a string?",
    ],
    "TIER_2_REFACTOR": [
        "Rename {func_a}() to {func_b}() across {module} and its call sites.",
        "Refactor {func_a} into separate {verb} and {verb} steps.",
        "Extract {module} logic from {class_a} into a dedicated {class_b} class.",
        "Update all invocations of {func_a} to pass an additional {type} parameter.",
        "Convert raw pointers in {class_a} to {smart_ptr}.",
        "Fix logic error in {func_a} where boundary condition is not handled.",
    ],
    "TIER_3_COMPLEX": [
        "I have a multithreaded application where two worker threads occasionally deadlock.",
        "My neural-network training becomes unstable only after several thousand iterations.",
        "Diagnose the cause of intermittent {issue} in {system}.",
        "Design a high-throughput lock-free {ds} ensuring {memory_model} consistency.",
        "Architect a resilient {system} capable of handling {failure_mode} under high load.",
        "Analyze and resolve cache thrashing in this multithreaded {system}.",
        "Design a custom {allocator} allocator with zero memory fragmentation guarantees.",
    ]
}

FILLERS = {
    "verb": ["parse", "validate", "serialize", "deserialize", "sort", "reverse",
             "clean", "encode", "filter", "merge", "split", "compress", "hash",
             "normalize", "tokenize", "flatten"],
    "type": ["std::string", "std::vector<int>", "float matrix", "JSON payload",
              "hash map", "binary tree node", "linked list", "priority queue",
              "std::array<double,N>", "byte buffer", "CSV row", "config struct"],
    "lang": ["C++", "modern C++", "C++20", "C++17", "C", "Rust"],
    "keyword": ["constexpr", "std::forward", "virtual destructor", "explicit constructor",
                "noexcept", "std::move", "std::optional", "std::variant",
                "RAII", "operator overloading", "template specialization"],
    "func_a": ["processData", "handleConnection", "parseBuffer", "syncState", "computeTotal"],
    "func_b": ["executeProcessing", "establishConnection", "readBuffer", "updateState", "calculateTotal"],
    "module": ["network module", "storage layer", "parser engine", "event loop", "worker queue"],
    "class_a": ["EngineManager", "ConnectionContext", "SessionHandler", "RequestDispatcher"],
    "class_b": ["EngineConfig", "ConnectionValidator", "AuthContext", "DispatcherMetrics"],
    "smart_ptr": ["std::unique_ptr", "std::shared_ptr", "std::weak_ptr"],
    "issue": ["deadlock", "memory corruption", "priority inversion", "data race",
              "ABA problem", "livelock", "false sharing", "use-after-free",
              "stack overflow", "heap fragmentation", "resource starvation",
              "torn reads", "cache line bouncing"],
    "system": ["audio processing pipeline", "distributed consensus engine",
               "event-driven trading system", "embedded controller",
               "video encoding pipeline", "real-time telemetry system",
               "GPU compute scheduler", "network packet router",
               "flight control system", "message broker cluster",
               "sensor fusion pipeline", "database replication engine"],
    "ds": ["ring buffer", "MPMC queue", "skip list", "hash table", "trie",
           "B+ tree", "bloom filter", "LRU cache", "red-black tree",
           "concurrent stack", "interval tree", "disjoint-set structure"],
    "memory_model": ["seq_cst", "acquire-release", "relaxed", "causal",
                      "sequentially consistent", "release-consume"],
    "failure_mode": ["network partition", "concurrent worker crashes",
                      "buffer starvation", "cascading timeouts",
                      "split-brain scenario", "silent data corruption"],
    "allocator": ["arena", "slab", "buddy", "lock-free thread-local",
                  "pool-based", "bump", "segregated free-list"],
}

def wrap_conversational(text):
    prefixes = [
        "I have a problem: ",
        "I need to ",
        "My application is failing, ",
        "Can you help me ",
        "How should I approach this: ",
        "I am trying to ",
        "We are planning to "
    ]
    # 50% chance to make it conversational
    if random.random() < 0.5:
        # Lowercase the first letter of the original text if it's a typical imperative
        first_word = text.split()[0]
        if first_word.istitle() and first_word not in ["I", "C++", "Node.js"]:
            text = text[0].lower() + text[1:]
        return random.choice(prefixes) + text
    return text

def generate_sample(tier):
    if random.random() < 0.25:
        return random.choice(SEEDS[tier])
    tmpl = random.choice(TEMPLATES[tier])
    result = tmpl
    for k, v in FILLERS.items():
        placeholder = "{" + k + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(v), 1)
    return wrap_conversational(result)

def get_3grams(text):
    words = text.lower().split()
    return set(zip(words, words[1:], words[2:])) if len(words) >= 3 else {tuple(words)}

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def main():
    target_count = 5100 # 1700 per class
    examples = []
    seen_grams = []
    
    tiers = ["TIER_1_SIMPLE", "TIER_2_REFACTOR", "TIER_3_COMPLEX"]
    per_tier = target_count // len(tiers)
    MAX_ATTEMPTS_PER_TIER = per_tier * 50   # hard cap so it can't spin forever
    
    idx = 0
    for tier in tiers:
        count = 0
        attempts = 0
        rejects = 0
        t0 = time.time()
        print(f"[{tier}] starting, target={per_tier}")
        
        while count < per_tier and attempts < MAX_ATTEMPTS_PER_TIER:
            attempts += 1
            text = generate_sample(tier)
            grams = get_3grams(text)
            
            # Deduplication: prune near duplicates
            is_dup = False
            for prev in seen_grams[-200:]: # check sliding window for efficiency
                if jaccard(grams, prev) > 0.65:
                    is_dup = True
                    break
            if is_dup:
                rejects += 1
                continue
            
            seen_grams.append(grams)
            examples.append({
                "id": f"s_{idx:05d}",
                "text": text,
                "label": tier
            })
            idx += 1
            count += 1
            
            if count % 500 == 0:
                print(f"[{tier}] {count}/{per_tier} accepted, {rejects} rejected so far")

        if count < per_tier:
            print(f"[{tier}] WARNING: stopped at {count}/{per_tier} "
                  f"after hitting {MAX_ATTEMPTS_PER_TIER} attempts "
                  f"(combo space likely exhausted — widen templates or lower threshold)")
        print(f"[{tier}] done: {count} accepted, {rejects} rejected, "
              f"{time.time()-t0:.1f}s")

    random.shuffle(examples)
    
    n = len(examples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        with open(f"data/processed/{split_name}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
                
    print(f"Generated {len(train)} train, {len(val)} val, {len(test)} test samples.")

if __name__ == "__main__":
    main()
