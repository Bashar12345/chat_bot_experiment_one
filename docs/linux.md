# Create a new directory
mkdir my_project

# Move into the directory
cd my_project

# Create a new file
touch notes.txt

# Rename the file
mv notes.txt project_notes.txt

# Move the file to another location
mv project_notes.txt ../

# Copy a file
cp ../project_notes.txt .

# Delete a file
rm project_notes.txt

# Delete a directory and its contents
rm -rf my_project





# Check internet connectivity
ping -c 5 google.com

# Find your system’s IP address
hostname -I

# Check open network ports
netstat -tulnp | grep LISTEN

# Get public IP
curl ifconfig.me


# Find all .log files modified in the last 7 days
find /var/log -name "*.log" -mtime -7

# Find and delete all empty files
find ~/Documents -type f -empty -delete

# Sort files by size in descending order
ls -lhS ~/Downloads

# Count the number of files in a directory
find ~/Projects -type f | wc -l

# Monitor real-time network traffic
sudo iftop -i eth0

# List all active connections
sudo ss -tulnp

# Scan open ports on your machine
sudo nmap -p- localhost

# Download a file in the background
wget -bqc http://example.com/largefile.zip



#!/bin/bash
SOURCE_DIR="$HOME/Documents"
BACKUP_DIR="$HOME/Backup"

mkdir -p "$BACKUP_DIR"
cp "$SOURCE_DIR"/*.txt "$BACKUP_DIR"

echo "Backup completed: $(date)" >> backup_log.txt

bash backup.sh




#!/bin/bash
# Cleanup Script

echo "Deleting old logs..."
find /var/log -type f -mtime +30 -delete

echo "Removing temporary files..."
rm -rf /tmp/* /var/tmp/*

echo "Clearing package cache..."
sudo apt autoremove -y && sudo apt autoclean -y

echo "System cleanup completed: $(date)" >> /var/log/cleanup.log

bash cleanup.sh


Open your crontab editor:

crontab -e

Add the following line to schedule a script (backup.sh) at 3 AM daily:

0 3 * * * /home/user/backup.sh >> /home/user/backup.log 2>&1

# Show real-time CPU and memory usage
top

# Display memory usage
free -h

# Get current CPU load
uptime



# Show the last 50 lines of system logs
tail -n 50 /var/log/syslog

# Find all failed login attempts
grep "Failed password" /var/log/auth.log

# Count the number of SSH login attempts
grep "Accepted password" /var/log/auth.log | wc -l



# Get weather information (replace CITY with your city)
curl -s "https://wttr.in/CITY?format=3"

# Fetch JSON data from an API
curl -s "https://api.github.com/users/octocat" | jq '.'








