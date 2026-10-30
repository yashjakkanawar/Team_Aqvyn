import faulthandler

# If the suite is still running after 90 s, print every thread's stack so a hang shows exactly where.
faulthandler.dump_traceback_later(90, exit=False)
