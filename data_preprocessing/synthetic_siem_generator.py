import pandas as pd
import random
from datetime import datetime, timedelta

def generate_siem_logs(num_records=10000, anomaly_ratio=0.05):
    users = [f"user_{i}" for i in range(50)]
    hosts = [f"host_{i}" for i in range(20)]
    processes = ["procA", "procB", "procC", "procD"]
    event_types = ["login", "file_access", "network_access", "privilege_change"]

    data = []
    start_time = datetime.now()

    for i in range(num_records):
        ts = start_time + timedelta(seconds=i * 5)
        user = random.choice(users)
        host = random.choice(hosts)
        proc = random.choice(processes)
        event = random.choice(event_types)

        is_anomaly = 1 if random.random() < anomaly_ratio else 0
        if is_anomaly:
            # Inject rare/unusual patterns
            user = f"mal_user_{random.randint(1, 5)}"
            event = "data_exfiltration"
            proc = "unknown_proc"

        data.append({
            "timestamp": ts,
            "user": user,
            "host": host,
            "process": proc,
            "event_type": event,
            "label": is_anomaly
        })

    df = pd.DataFrame(data)
    df.to_csv("data/synthetic_siem_logs.csv", index=False)
    print("Synthetic SIEM logs generated at: data/synthetic_siem_logs.csv")

if __name__ == '__main__':
    generate_siem_logs()

