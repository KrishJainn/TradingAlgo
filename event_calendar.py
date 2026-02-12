"""
Event Calendar — Scheduled market events for exit signal tightening.

Tracks known volatility events (RBI MPC, Union Budget, F&O expiry, US Fed,
earnings season, etc.) and provides an exit-tightening signal that the
SmartExitEngine uses to protect profitable positions before predictable
volatility windows.

Trading-day awareness
=====================
Proximity calculations use **trading days** (Mon–Fri), not calendar days.
This prevents the "weekend blind spot" where a Monday event with
look_ahead_days=1 would be invisible on Friday (3 calendar days away)
even though Friday is the last trading day the system can act on.

    Friday → Monday  = 1 trading day  (not 3 calendar days)
    Friday → Tuesday = 2 trading days (not 4 calendar days)

Limitation: Indian market holidays (Republic Day, Diwali, etc.) are NOT
accounted for — only weekends are skipped.  A position held over a
3-day holiday weekend would see the same blind-spot as a regular
weekend.  This is acceptable for the 2025-2026 trading period because
the look_ahead_days values are conservative enough to absorb 1-day
holiday gaps.

Usage
=====
    from event_calendar import EventCalendar

    cal = EventCalendar()

    # What events are coming in the next 5 trading days?
    events = cal.get_upcoming_events(date(2025, 4, 7), look_ahead_days=5)

    # How much should I tighten exits? (0.0 = normal, 1.0 = max)
    factor = cal.get_exit_tightening_factor(date(2025, 4, 7))

    # Full signal score considering P&L
    signal = cal.exit_signal_score(date(2025, 4, 7), position_pnl_pct=0.025)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Event data structure
# ═══════════════════════════════════════════════════════════════════════════════

class MarketEvent(NamedTuple):
    """A single scheduled market event."""
    date: date
    event_type: str       # category key
    name: str             # human-readable name
    severity: float       # 0.0–1.0 (how much it can move markets)
    look_ahead_days: int  # how many days before the event to start tightening


# ═══════════════════════════════════════════════════════════════════════════════
# EventCalendar
# ═══════════════════════════════════════════════════════════════════════════════

class EventCalendar:
    """Scheduled market events for 2025–2026 India + US.

    Provides exit-tightening signals based on proximity to known
    volatility events.  Events are hardcoded for the current trading
    period (2025-01 through 2026-12).
    """

    def __init__(self) -> None:
        self._events: List[MarketEvent] = []
        self._build_events()
        # Sort by date for efficient lookup
        self._events.sort(key=lambda e: e.date)
        logger.info(f"[EventCalendar] Loaded {len(self._events)} events (2025-2026)")

    # ── Event database builder ─────────────────────────────────────────

    def _build_events(self) -> None:
        """Build the complete event database for 2025-2026."""
        self._add_rbi_mpc()
        self._add_union_budget()
        self._add_fo_monthly_expiry()
        self._add_fo_weekly_expiry()
        self._add_us_fed_fomc()
        self._add_us_macro_data()
        self._add_india_macro_data()
        self._add_earnings_season()

    # ── RBI Monetary Policy Committee ──────────────────────────────────

    def _add_rbi_mpc(self) -> None:
        """RBI MPC meetings — 6 per year, decision on last day.

        Severity: 0.8 (can move NIFTY 1-2%)
        Look-ahead: 2 days before first day of meeting
        """
        # 2025 RBI MPC dates (decision date = last day)
        rbi_2025 = [
            (date(2025, 2, 7), "RBI MPC Feb 2025"),
            (date(2025, 4, 9), "RBI MPC Apr 2025"),
            (date(2025, 6, 6), "RBI MPC Jun 2025"),
            (date(2025, 8, 8), "RBI MPC Aug 2025"),
            (date(2025, 10, 8), "RBI MPC Oct 2025"),
            (date(2025, 12, 5), "RBI MPC Dec 2025"),
        ]
        # 2026 RBI MPC dates
        rbi_2026 = [
            (date(2026, 2, 6), "RBI MPC Feb 2026"),
            (date(2026, 4, 8), "RBI MPC Apr 2026"),
            (date(2026, 6, 5), "RBI MPC Jun 2026"),
            (date(2026, 8, 7), "RBI MPC Aug 2026"),
            (date(2026, 10, 9), "RBI MPC Oct 2026"),
            (date(2026, 12, 4), "RBI MPC Dec 2026"),
        ]
        for dt, name in rbi_2025 + rbi_2026:
            self._events.append(MarketEvent(
                date=dt, event_type="rbi_mpc", name=name,
                severity=0.8, look_ahead_days=2,
            ))

    # ── Union Budget ───────────────────────────────────────────────────

    def _add_union_budget(self) -> None:
        """Union Budget — presented Feb 1 each year.

        Severity: 1.0 (can move NIFTY 2-5%)
        Look-ahead: 3 days
        """
        for year in (2025, 2026):
            self._events.append(MarketEvent(
                date=date(year, 2, 1),
                event_type="union_budget",
                name=f"Union Budget {year}",
                severity=1.0,
                look_ahead_days=3,
            ))

    # ── F&O Monthly Expiry ─────────────────────────────────────────────

    def _add_fo_monthly_expiry(self) -> None:
        """F&O monthly expiry — last Thursday of each month.

        Severity: 0.4 (increased vol from rollover)
        Look-ahead: 1 day
        """
        for year in (2025, 2026):
            for month in range(1, 13):
                last_thu = self._last_thursday(year, month)
                self._events.append(MarketEvent(
                    date=last_thu,
                    event_type="fo_monthly_expiry",
                    name=f"F&O Monthly Expiry {last_thu.strftime('%b %Y')}",
                    severity=0.4,
                    look_ahead_days=1,
                ))

    # ── F&O Weekly Expiry ──────────────────────────────────────────────

    def _add_fo_weekly_expiry(self) -> None:
        """F&O weekly expiry — every Thursday.

        Severity: 0.2 (minor vol bump)
        Look-ahead: 0 days (only on the day itself)
        """
        # Generate every Thursday from Jan 2025 to Dec 2026
        start = date(2025, 1, 2)  # first Thursday of 2025
        # Find first Thursday
        while start.weekday() != 3:  # 3 = Thursday
            start += timedelta(days=1)

        current = start
        end = date(2026, 12, 31)
        while current <= end:
            # Skip if it's a monthly expiry (already covered with higher severity)
            last_thu = self._last_thursday(current.year, current.month)
            if current != last_thu:
                self._events.append(MarketEvent(
                    date=current,
                    event_type="fo_weekly_expiry",
                    name=f"F&O Weekly Expiry {current.isoformat()}",
                    severity=0.2,
                    look_ahead_days=0,
                ))
            current += timedelta(days=7)

    # ── US Federal Reserve FOMC ────────────────────────────────────────

    def _add_us_fed_fomc(self) -> None:
        """US Fed FOMC meetings — 8 per year, decision on day 2.

        Severity: 0.6 (US rate decisions affect FII flows to India)
        Look-ahead: 1 day
        """
        # 2025 FOMC decision dates (day 2 of each meeting)
        fomc_2025 = [
            (date(2025, 1, 29), "FOMC Jan 2025"),
            (date(2025, 3, 19), "FOMC Mar 2025"),
            (date(2025, 5, 7), "FOMC May 2025"),
            (date(2025, 6, 18), "FOMC Jun 2025"),
            (date(2025, 7, 30), "FOMC Jul 2025"),
            (date(2025, 9, 17), "FOMC Sep 2025"),
            (date(2025, 10, 29), "FOMC Oct 2025"),
            (date(2025, 12, 10), "FOMC Dec 2025"),
        ]
        # 2026 FOMC decision dates
        fomc_2026 = [
            (date(2026, 1, 28), "FOMC Jan 2026"),
            (date(2026, 3, 18), "FOMC Mar 2026"),
            (date(2026, 5, 6), "FOMC May 2026"),
            (date(2026, 6, 17), "FOMC Jun 2026"),
            (date(2026, 7, 29), "FOMC Jul 2026"),
            (date(2026, 9, 16), "FOMC Sep 2026"),
            (date(2026, 10, 28), "FOMC Oct 2026"),
            (date(2026, 12, 9), "FOMC Dec 2026"),
        ]
        for dt, name in fomc_2025 + fomc_2026:
            self._events.append(MarketEvent(
                date=dt, event_type="us_fomc", name=name,
                severity=0.6, look_ahead_days=1,
            ))

    # ── US Macro Data (CPI, NFP) ──────────────────────────────────────

    def _add_us_macro_data(self) -> None:
        """US CPI and NFP releases — monthly.

        CPI: ~10th-13th of each month. Severity: 0.5
        NFP: first Friday of each month.   Severity: 0.5
        Look-ahead: 1 day for both
        """
        # Approximate US CPI release dates (typically mid-month)
        for year in (2025, 2026):
            for month in range(1, 13):
                # CPI: typically 10th-15th, use 12th as proxy
                cpi_day = min(12, self._max_day(year, month))
                cpi_date = date(year, month, cpi_day)
                # Adjust to weekday
                while cpi_date.weekday() > 4:
                    cpi_date += timedelta(days=1)
                self._events.append(MarketEvent(
                    date=cpi_date,
                    event_type="us_cpi",
                    name=f"US CPI {cpi_date.strftime('%b %Y')}",
                    severity=0.5,
                    look_ahead_days=1,
                ))

                # NFP: first Friday of each month
                nfp = date(year, month, 1)
                while nfp.weekday() != 4:  # 4 = Friday
                    nfp += timedelta(days=1)
                self._events.append(MarketEvent(
                    date=nfp,
                    event_type="us_nfp",
                    name=f"US NFP {nfp.strftime('%b %Y')}",
                    severity=0.5,
                    look_ahead_days=1,
                ))

    # ── India Macro Data (GDP, IIP) ────────────────────────────────────

    def _add_india_macro_data(self) -> None:
        """India GDP and IIP data releases — quarterly/monthly.

        GDP: end of Feb, May, Aug, Nov (for previous quarter)
        IIP: monthly, around 12th of each month
        Severity: 0.4, Look-ahead: 1 day
        """
        for year in (2025, 2026):
            # GDP quarterly (approximate release dates)
            gdp_months = [2, 5, 8, 11]
            for m in gdp_months:
                gdp_date = date(year, m, 28)
                while gdp_date.weekday() > 4:
                    gdp_date -= timedelta(days=1)
                self._events.append(MarketEvent(
                    date=gdp_date,
                    event_type="india_gdp",
                    name=f"India GDP Q{(m // 3)} {year}",
                    severity=0.4,
                    look_ahead_days=1,
                ))

    # ── Quarterly Earnings Season ──────────────────────────────────────

    def _add_earnings_season(self) -> None:
        """Quarterly earnings season start dates.

        Companies start reporting results:
        - Q3 results: mid-January
        - Q4 results: mid-April
        - Q1 results: mid-July
        - Q2 results: mid-October

        Severity: 0.5 (stock-specific but affects overall sentiment)
        Look-ahead: 1 day
        """
        for year in (2025, 2026):
            earnings = [
                (date(year, 1, 15), f"Q3 Earnings Season Start {year}"),
                (date(year, 4, 15), f"Q4 Earnings Season Start {year}"),
                (date(year, 7, 15), f"Q1 Earnings Season Start {year}"),
                (date(year, 10, 15), f"Q2 Earnings Season Start {year}"),
            ]
            for dt, name in earnings:
                # Adjust to weekday
                while dt.weekday() > 4:
                    dt += timedelta(days=1)
                self._events.append(MarketEvent(
                    date=dt,
                    event_type="earnings_season",
                    name=name,
                    severity=0.5,
                    look_ahead_days=1,
                ))

    # ── Helper methods ─────────────────────────────────────────────────

    @staticmethod
    def _last_thursday(year: int, month: int) -> date:
        """Find the last Thursday of a given month."""
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        # Walk back to Thursday (weekday 3)
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        return last_day

    @staticmethod
    def _max_day(year: int, month: int) -> int:
        """Return the last day of a month."""
        if month == 12:
            return 31
        return (date(year, month + 1, 1) - timedelta(days=1)).day

    @staticmethod
    def _trading_days_between(from_date: date, to_date: date) -> int:
        """Count trading days (Mon–Fri) between two dates, exclusive of from_date.

        Examples:
            Friday → Monday   = 1 trading day
            Friday → Tuesday  = 2 trading days
            Thursday → Friday = 1 trading day
            Monday → Monday   = 0 trading days (same day)

        Does NOT account for Indian market holidays — only weekends.
        """
        if to_date <= from_date:
            return 0
        count = 0
        current = from_date + timedelta(days=1)
        while current <= to_date:
            if current.weekday() < 5:  # Mon=0 .. Fri=4
                count += 1
            current += timedelta(days=1)
        return count

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def get_upcoming_events(
        self,
        check_date: date,
        look_ahead_days: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get all events within N trading days of check_date.

        Uses trading days (Mon–Fri) so Friday correctly sees Monday
        events as 1 day away, not 3.

        Parameters
        ----------
        check_date : date
            The date to check from.
        look_ahead_days : int
            How many trading days ahead to look.

        Returns
        -------
        list of dict
            Each dict: {date, event_type, name, severity, look_ahead_days,
                        days_until (trading days), calendar_days_until}
        """
        # Convert trading-day window to a generous calendar-day cutoff
        # (look_ahead_days trading days ≤ look_ahead_days * 2 calendar days)
        cutoff = check_date + timedelta(days=look_ahead_days * 2 + 2)
        results = []
        for evt in self._events:
            if check_date <= evt.date <= cutoff:
                trading_days = self._trading_days_between(check_date, evt.date)
                if trading_days > look_ahead_days:
                    continue
                results.append({
                    "date": evt.date.isoformat(),
                    "event_type": evt.event_type,
                    "name": evt.name,
                    "severity": evt.severity,
                    "look_ahead_days": evt.look_ahead_days,
                    "days_until": trading_days,
                    "calendar_days_until": (evt.date - check_date).days,
                })
        return results

    def get_exit_tightening_factor(self, check_date: date) -> float:
        """Compute how much to tighten exits based on upcoming events.

        Uses **trading days** (Mon–Fri) for proximity so that Friday
        correctly sees Monday events as 1 trading day away.

        For each upcoming event within its own look_ahead window:
            proximity_factor = 1.0 - (trading_days_until / look_ahead_days)
            event_score = severity * proximity_factor

        Returns the max event_score across all events (never summed).

        Parameters
        ----------
        check_date : date
            The current trading date.

        Returns
        -------
        float
            0.0 = no events nearby (normal exits)
            1.0 = major event imminent (maximum tightening)
        """
        max_score = 0.0

        for evt in self._events:
            calendar_days = (evt.date - check_date).days

            # Only consider events in the future or today
            if calendar_days < 0:
                continue

            # Quick skip: if calendar days > look_ahead * 3, no chance
            if calendar_days > evt.look_ahead_days * 3 + 3:
                continue

            # Compute trading days (the actual distance that matters)
            trading_days = self._trading_days_between(check_date, evt.date)

            # Only tighten within the event's own look_ahead window
            if trading_days > evt.look_ahead_days:
                continue

            # Proximity ramp: 1.0 on event day, decreasing toward boundary.
            # Uses (look_ahead + 1) as denominator so the outermost day
            # still gets a meaningful signal instead of 0.0:
            #   look_ahead=1, td=1 → 0.50  (was 0.0 with old formula)
            #   look_ahead=1, td=0 → 1.00
            #   look_ahead=2, td=2 → 0.33  (was 0.0)
            if evt.look_ahead_days == 0:
                if trading_days == 0:
                    proximity = 1.0
                else:
                    continue
            else:
                proximity = 1.0 - (trading_days / (evt.look_ahead_days + 1))

            event_score = evt.severity * proximity
            max_score = max(max_score, event_score)

        return min(1.0, max_score)

    def exit_signal_score(
        self,
        check_date: date,
        position_pnl_pct: float,
    ) -> float:
        """Compute exit signal score incorporating P&L context.

        Profitable positions get stronger tightening (protect gains),
        losing positions get moderate tightening (event could help),
        flat positions get mild tightening.

        Parameters
        ----------
        check_date : date
            The current trading date.
        position_pnl_pct : float
            Current unrealised P&L as a fraction (e.g. 0.02 = +2%).

        Returns
        -------
        float
            Signal score in [0.0, 1.0] for SmartExitEngine weighted voting.
        """
        tightening = self.get_exit_tightening_factor(check_date)

        if tightening == 0.0:
            return 0.0

        # In profit > 1%: strong signal to protect
        if position_pnl_pct > 0.01:
            return tightening * 0.8

        # In loss > 0.5%: moderate — event could go either way
        if position_pnl_pct < -0.005:
            return tightening * 0.5

        # Flat or small P&L: mild tightening
        return tightening * 0.3

    def get_event_context_for_prompt(
        self,
        check_date: date,
        look_ahead_days: int = 5,
    ) -> str:
        """Build a compact text block for the Gemini exit prompt.

        Returns empty string if no events upcoming.
        """
        events = self.get_upcoming_events(check_date, look_ahead_days)
        if not events:
            return ""

        tightening = self.get_exit_tightening_factor(check_date)

        lines = [f"Exit tightening factor: {tightening:.2f}"]
        for evt in events:
            lines.append(
                f"  {evt['name']} (in {evt['days_until']}d, "
                f"severity={evt['severity']:.1f})"
            )

        return "\n".join(lines)

    # ── Diagnostics ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return summary stats about the event calendar."""
        type_counts: Dict[str, int] = {}
        for evt in self._events:
            type_counts[evt.event_type] = type_counts.get(evt.event_type, 0) + 1
        return {
            "total_events": len(self._events),
            "event_types": type_counts,
            "date_range": f"{self._events[0].date} to {self._events[-1].date}",
        }
