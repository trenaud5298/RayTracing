#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <type_traits>


// Timer class template only works for std::chrono::duration types
template <typename DurationType>
class Timer {
static_assert(
    std::is_base_of<std::chrono::duration<typename DurationType::rep, typename DurationType::period>, DurationType>::value,
    "Timer can only be used with std::chrono::duration types such as seconds, milliseconds, etc."
);


public:
    Timer(const std::string& name = "Timer") : 
    m_timerName(name), m_startTime(std::chrono::high_resolution_clock::now()) {
        //Timer Begins On Initialization
    }

    ~Timer() {
        std::cout<<m_timerName<<" Total Elapsed Time: "<<getElapsedTimeAsString()<<"\n";
    }

    std::string getElapsedTimeAsString() const {
        return std::to_string(getElapsedTime().count()) + " " + getUnitName();
    }
    
    DurationType getElapsedTime() const {
        // Casts the duration to the specified DurationType.
        return std::chrono::duration_cast<DurationType>(std::chrono::high_resolution_clock::now() - m_startTime);
    }

    std::string getUnitName() const {
        if (std::is_same<DurationType, std::chrono::seconds>::value) {
            return "seconds";
        } else if (std::is_same<DurationType, std::chrono::milliseconds>::value) {
            return "milliseconds";
        } else if (std::is_same<DurationType, std::chrono::microseconds>::value) {
            return "microseconds";
        } else if (std::is_same<DurationType, std::chrono::nanoseconds>::value) {
            return "nanoseconds";
        }
        return "unknown";
    }
    


private:
    std::string m_timerName;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};


























#include <iostream>
#include <chrono>
#include <string>

template <typename DurationType = std::chrono::seconds>
class TimerOld {
public:
    // Constructor takes an optional name for the timer.
    TimerOld(const std::string& name = "Timer") : name(name), start(std::chrono::high_resolution_clock::now()) {}

    // Destructor will print the time in the specified unit when the timer goes out of scope.
    ~TimerOld() {
        if (!stopped) {
            stop();
        }
    }

    // Stop the timer manually, and print the elapsed time in the specified unit.
    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        auto durationInRequestedUnit = std::chrono::duration_cast<DurationType>(duration);

        std::cout << name << " elapsed time: " << durationInRequestedUnit.count() << " " << getUnitName() << "\n";
        stopped = true;
    }

    // Get the elapsed time without stopping the timer.
    double getElapsedTime() const {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        auto durationInRequestedUnit = std::chrono::duration_cast<DurationType>(duration);
        return durationInRequestedUnit.count();
    }

private:
    // Get the string representation of the time unit.
    std::string getUnitName() const {
        if (std::is_same<DurationType, std::chrono::seconds>::value) {
            return "seconds";
        } else if (std::is_same<DurationType, std::chrono::milliseconds>::value) {
            return "milliseconds";
        } else if (std::is_same<DurationType, std::chrono::microseconds>::value) {
            return "microseconds";
        } else if (std::is_same<DurationType, std::chrono::nanoseconds>::value) {
            return "nanoseconds";
        }
        return "unknown";
    }

    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    bool stopped = false;
};



#endif