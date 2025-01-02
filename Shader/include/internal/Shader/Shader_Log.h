#ifndef SHADER_LOG_H
#define SHADER_LOG_H

#ifndef LOG_TYPE_ERROR
    #define LOG_TYPE_ERROR    0x01
#endif

#ifndef LOG_TYPE_WARNING
    #define LOG_TYPE_WARNING  0x02
#endif

#ifndef LOG_TYPE_INFO
    #define LOG_TYPE_INFO     0x04
#endif

#ifndef LOG_TYPE_DEBUG
    #define LOG_TYPE_DEBUG    0x08
#endif

#ifndef LOG_TYPE_ALL
    #define LOG_TYPE_ALL (LOG_TYPE_ERROR | LOG_TYPE_WARNING | LOG_TYPE_INFO | LOG_TYPE_DEBUG)
#endif

#ifndef STARTING_LOG_SIZE
    #define STARTING_LOG_SIZE 1024
#endif

#include <iostream>
#include <vector>

/**
 * @file Shader_Log.h
 * @brief This Struct Represents A Single Log Message That Is Created
 * And Stored In The ShaderLog Class. This Is Intended To Only Be Used
 * In Internal Shader Library Use, However If Needed Can Be Manually 
 * Included For Use Elsewhere, And Is Header Only.
 */
struct LogEntry {
    uint8_t type;
    std::string message;

    LogEntry() : type(0), message("") {}
    LogEntry( uint8_t type, const std::string& message ) : type(type), message(message) {}
    
    std::string toString(bool addNewLineChar = false) {
        std::string result = "";
        if(type & LOG_TYPE_ERROR   ) { result.append("[ERROR]");   }
        if(type & LOG_TYPE_WARNING ) { result.append("[WARNING]"); }
        if(type & LOG_TYPE_INFO    ) { result.append("[INFO]");    }
        if(type & LOG_TYPE_DEBUG   ) { result.append("[DEBUG]");   }
        result.append(" ");
        result.append(message);
        if(addNewLineChar) { result.append("\n"); }
        return result;
    }

    operator bool() {
        return (type==0);
    }
};


/**
 * This Class Represents A Simple Wrapper That Holds A
 * Vector Of Log Messages And Provides Basic Functionality
 * To Get Specific Log Messages, Convert Them To Strings
 * And Also Get A String Version Of The Entire Log Or
 * Of Just A Specificed Type
 */
class ShaderLog {
//Methods
public:
    /**
     * Constryctor For Shader Log That Resevers The Staring Log Size For
     * Internal Vector And Creates Inital Log Message, Reporting That Log
     * Has Been Created.
     */
    ShaderLog() {
        m_log.reserve(STARTING_LOG_SIZE);
        m_log.push_back(LogEntry(LOG_TYPE_INFO, "Log Created"));
    }

    /**
     * This Destructor Handles The Cleanup Of The Shader Log
     * Class And Optionally Saves The Shader Log As A .txt
     * File If The Method `saveUponDestruction()` Has Been Called
     * Before Destruction.
     */
    ~ShaderLog() {
        //TODO: Add Saving Implementation
        if(m_saveUponDestruction) {
            std::cerr<<"TODO: Add Save Feature"<<std::endl;
        }
    }

    /**
     * Method To Add A New Log Message To Log.
     * 
     * This Method Creates A New `LogEntry` With
     * Passed In Parameters And Pushes It To The
     * Back Of The LogEntry Vector
     */
    void addEntry(uint8_t logType, const std::string& logMessage) {
        m_log.push_back( LogEntry(logType, logMessage) );
    }

    /**
     * Method That Adds A New LogEntry To Log
     * 
     * Pushes The Passed In LogEntry To The Back
     * Of The LogEntry Vector
     */
    void addEntry(const LogEntry& newEntry) {
        m_log.push_back( newEntry );
    }

    /**
     * Returns The Last LogEntry Object That Mathces
     * The Passed In logType From The LogEntry Vector
     * 
     * @param logType Bitmap Representing The Type Of Log To Search For In Log.
     * Allows `|` Of Multiple Log Types To Search For The Last Entry That Matches
     * At Least One Of The Specified Types
     * =============================================================================
     * `Available Log Types`
     *  
     * - `LOG_TYPE_ERROR`
     * 
     *  - `LOG_TYPE_WARNING`
     * 
     *  - `LOG_TYPE_INFO`
     * 
     *  - `LOG_TYPE_DEBUG`
     */
    LogEntry getLastEntryOfType(uint8_t logType) {
        for(int logIndex = m_log.size()-1; logIndex >= 0; logIndex--) {
            if(m_log.at(logIndex).type & logType) {
                return m_log.at(logIndex);
            }
        }
        return LogEntry();
    }

    /**
     * Returns The Last LogEntry Object That Mathces
     * The Passed In logType From The LogEntry Vector
     * Converted Into A String Representation.
     * 
     * @param logType Bitmap Representing The Type Of Log To Search For In Log.
     * Allows `|` Of Multiple Log Types To Search For The Last Entry That Matches
     * At Least One Of The Specified Types
     * =============================================================================
     * `Available Log Types`
     *  
     * - `LOG_TYPE_ERROR`
     * 
     *  - `LOG_TYPE_WARNING`
     * 
     *  - `LOG_TYPE_INFO`
     * 
     *  - `LOG_TYPE_DEBUG`
     */
    std::string getLastEntryOfTypeAsString(uint8_t logType) {
        return getLastEntryOfType(logType).toString();
    }
    
    /**
     * Returns The First LogEntry Object That Mathces
     * The Passed In logType From The LogEntry Vector
     * 
     * @param logType Bitmap Representing The Type Of Log To Search For In Log.
     * Allows `|` Of Multiple Log Types To Search For The Last Entry That Matches
     * At Least One Of The Specified Types
     * =============================================================================
     * `Available Log Types`
     *  
     * - `LOG_TYPE_ERROR`
     * 
     *  - `LOG_TYPE_WARNING`
     * 
     *  - `LOG_TYPE_INFO`
     * 
     *  - `LOG_TYPE_DEBUG`
     */
    LogEntry getFirstEntryOfType(uint8_t logType) {
        for(int logIndex = 0; logIndex < m_log.size(); logIndex++) {
            if(m_log.at(logIndex).type & logType) {
                return m_log.at(logIndex);
            }
        }
        return LogEntry();
    }

    /**
     * Returns The First LogEntry Object That Mathces
     * The Passed In logType From The LogEntry Vector
     * Converted Into A String Representation
     * 
     * @param logType Bitmap Representing The Type Of Log To Search For In Log.
     * Allows `|` Of Multiple Log Types To Search For The Last Entry That Matches
     * At Least One Of The Specified Types
     * =============================================================================
     * `Available Log Types`
     *  
     * - `LOG_TYPE_ERROR`
     * 
     *  - `LOG_TYPE_WARNING`
     * 
     *  - `LOG_TYPE_INFO`
     * 
     *  - `LOG_TYPE_DEBUG`
     */
    std::string getFirstEntryOfTypeAsString(uint8_t logType) {
        return getFirstEntryOfType(logType).toString();
    }

    /**
     * Returns A String Representation Of All LogEntrys
     * That Match The Passed In Log Type
     * 
     * @param logType Bitmap Representing The Type Of Log To Search For In Log.
     * Allows `|` Of Multiple Log Types To Search For The Last Entry That Matches
     * At Least One Of The Specified Types
     * =============================================================================
     * `Available Log Types`
     *  
     * - `LOG_TYPE_ERROR`
     * 
     *  - `LOG_TYPE_WARNING`
     * 
     *  - `LOG_TYPE_INFO`
     * 
     *  - `LOG_TYPE_DEBUG`
     */
    std::string getLogOfTypeAsString(uint8_t logType) {
        std::string result = "--Start Of Shader Library Log--\n";
        for(int logIndex = 0; logIndex < m_log.size(); logIndex++) {
            if(m_log.at(logIndex).type & logType) {
                result.append( m_log.at(logIndex).toString(true) );
            }
        }
        result.append("---End Of Shader Library Log---");
        return result;
    }

    /**
     * Method That Sets The Log To Save To A .txt
     * File Automatically Upon Deletion. Allows
     * The User To Specify A Specific Name
     * Or Use The Default
     */
    void saveUponDestruction(std::string fileName = "log") {
        //TODO: Add The Ability To Set File Name Or Leave Default
        m_saveUponDestruction = true;
    }

private:
    std::vector<LogEntry> m_log;
    bool m_saveUponDestruction = false;
};

#endif