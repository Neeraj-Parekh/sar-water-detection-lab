"""
SAR Water Detection - QA and Audit Module
==========================================

Traffic Light QA system and audit logging.
"""

import json
from pathlib import Path
from datetime import datetime

AUDIT_FILE = Path('./audit_log.json')
GOLDEN_SET_FILE = Path('./golden_set.json')


def load_audit_log():
    """Load audit log."""
    if AUDIT_FILE.exists():
        with open(AUDIT_FILE, 'r') as f:
            return json.load(f)
    return {'chips': {}, 'created': datetime.now().isoformat()}


def save_audit_log(audit):
    """Save audit log."""
    audit['updated'] = datetime.now().isoformat()
    with open(AUDIT_FILE, 'w') as f:
        json.dump(audit, f, indent=2)


def load_golden_set():
    """Load golden set list."""
    if GOLDEN_SET_FILE.exists():
        with open(GOLDEN_SET_FILE, 'r') as f:
            return json.load(f)
    return {'chips': [], 'created': datetime.now().isoformat()}


def save_golden_set(golden):
    """Save golden set."""
    golden['updated'] = datetime.now().isoformat()
    with open(GOLDEN_SET_FILE, 'w') as f:
        json.dump(golden, f, indent=2)


def set_chip_status(chip_name, status, notes=''):
    """
    Set chip QA status.
    
    Args:
        chip_name: Name of the chip
        status: 'green' (perfect), 'yellow' (ambiguous), 'red' (corrupted)
        notes: Optional notes
    """
    audit = load_audit_log()
    
    audit['chips'][chip_name] = {
        'status': status,
        'notes': notes,
        'timestamp': datetime.now().isoformat()
    }
    
    save_audit_log(audit)
    
    # Update golden set
    golden = load_golden_set()
    
    if status == 'green':
        if chip_name not in golden['chips']:
            golden['chips'].append(chip_name)
    else:
        if chip_name in golden['chips']:
            golden['chips'].remove(chip_name)
    
    save_golden_set(golden)
    
    return True


def get_chip_status(chip_name):
    """Get chip QA status."""
    audit = load_audit_log()
    return audit.get('chips', {}).get(chip_name, {'status': 'none'})


def get_status_counts():
    """Get count of chips by status."""
    audit = load_audit_log()
    counts = {'green': 0, 'yellow': 0, 'red': 0, 'none': 0}
    
    for chip_info in audit.get('chips', {}).values():
        status = chip_info.get('status', 'none')
        counts[status] = counts.get(status, 0) + 1
    
    return counts


def get_golden_set():
    """Get list of golden set chips."""
    golden = load_golden_set()
    return golden.get('chips', [])


def export_audit_report():
    """Export audit report as summary."""
    audit = load_audit_log()
    golden = load_golden_set()
    counts = get_status_counts()
    
    report = {
        'summary': {
            'total_reviewed': len(audit.get('chips', {})),
            'green': counts['green'],
            'yellow': counts['yellow'],
            'red': counts['red'],
            'golden_set_size': len(golden.get('chips', []))
        },
        'golden_set': golden.get('chips', []),
        'all_chips': audit.get('chips', {}),
        'generated': datetime.now().isoformat()
    }
    
    report_file = Path('./audit_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_file


# Window-level voting

WINDOW_VOTES_FILE = Path('./window_votes.json')


def load_window_votes():
    """Load window votes."""
    if WINDOW_VOTES_FILE.exists():
        with open(WINDOW_VOTES_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_window_votes(votes):
    """Save window votes."""
    with open(WINDOW_VOTES_FILE, 'w') as f:
        json.dump(votes, f, indent=2)


def set_window_vote(chip_name, window_idx, vote):
    """
    Set vote for a specific window.
    
    Args:
        chip_name: Chip name
        window_idx: Window index (0-14)
        vote: 'green', 'red', or 'none'
    """
    votes = load_window_votes()
    
    if chip_name not in votes:
        votes[chip_name] = {}
    
    votes[chip_name][str(window_idx)] = {
        'vote': vote,
        'timestamp': datetime.now().isoformat()
    }
    
    save_window_votes(votes)


def get_window_votes(chip_name):
    """Get all window votes for a chip."""
    votes = load_window_votes()
    return votes.get(chip_name, {})


def get_accepted_windows(chip_name):
    """Get list of window indices that are voted green."""
    votes = get_window_votes(chip_name)
    accepted = []
    for idx, info in votes.items():
        if info.get('vote') == 'green':
            accepted.append(int(idx))
    return accepted
