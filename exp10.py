# ======================== CSV FILE: WHAT YOU HAVE TO DO ========================
# 1. In VS Code, create a NEW FILE in the SAME folder as this .py file.
# 2. Name it  exp10.csv
# 3. Copy the lines below into it (WITHOUT the '# ' at the start), save with Ctrl+S.
#
#    row_id,col_0,col_1,col_2,col_3
#    R0,2,4,1,5
#    R1,3,0,2,1
#    R2,1,2,3,4
#    R3,5,1,0,2
#
# RULES: EXACTLY 4 rows (R0, R1, R2, R3), whole numbers only.
#
# 4. Change DOMAIN below to your chit's domain.   5. Run:  python exp10.py
# ===============================================================================

# MPI MATRIX MULTIPLICATION (simulated with threads)    C = A x B
#   BROADCAST : master sends matrix B to every worker
#   SCATTER   : master gives each worker one row of A
#   each worker calculates its row of C (all at the same time)
#   GATHER    : master collects all the rows into C

import csv
import threading

DOMAIN = "Predictive Distributed Task Scheduler using ML"   # change to your chit
FILE = "exp10.csv"

# ---------- read matrix A from the CSV ----------
A = []
for row in csv.DictReader(open(FILE)):
    A.append([int(row["col_0"]), int(row["col_1"]), int(row["col_2"]), int(row["col_3"])])

B = A                  # multiply A by itself
C = [0, 0, 0, 0]       # 4 empty slots for the 4 answer rows


# ---------- what each worker does: calculate row i of C ----------
def worker(i):
    answer = []
    for j in range(4):                       # for each column of B
        total = 0
        for k in range(4):
            total = total + A[i][k] * B[k][j]
        answer.append(total)
    C[i] = answer                            # put the row in its own slot


print("=== MPI Matrix Multiplication for", DOMAIN, "===")
print()
print("Matrix A:")
for row in A:
    print("  ", row)

print()
print("BROADCAST: Master sends matrix B to all 4 workers")
print("SCATTER:")
for i in range(4):
    print("   Master sends row", i, A[i], "to Worker", i)

# ---------- start 4 workers at the same time ----------
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()           # wait until every worker has finished

# ---------- gather ----------
print()
print("GATHER: Master collects the rows from the workers")
for i in range(4):
    print("   Worker", i, "sent back row", i, ":", C[i])

print()
print("Final matrix C = A x B:")
for row in C:
    print("  ", row)