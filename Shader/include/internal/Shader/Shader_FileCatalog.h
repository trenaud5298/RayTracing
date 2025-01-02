#ifndef FILETREE_H
#define FILETREE_H

#include <vector>
#include <set>
#include <filesystem>


/**
 * This Enum Type Allows For The Use Of 
 */
enum SearchPointType : uint8_t {
    DIRECTORY = 1,
    INDIVIDUAL_PATH = 2
};

/**
 * @file Shader_FileCatalog.h
 * @struct SearchPoint
 * 
 * @brief The SearchPoint Struct Represents A Single Search Point Used By The FileCatalog Class
 * To Define The Scope Of Resource Searching. 
 * 
 * A Search Point Can Be Either A Directory Or An Individual File Path, As Defined By The `type`
 * Field. Additionally, For Directory Search Points, The `maxRecursiveDepth` Field Specifies How
 * Deeply To Search Within Sub-Directories. This Struct Serves As A Building Block For The
 * FileCatalog's Functionality, Allowing Precise Control Over The Scope And Method Of Resource
 * Fetching In The Shader::ResourceManager Sub-Module.
 */
struct SearchPoint {
    SearchPointType m_type;
    uint8_t m_maxRecursiveDepth;
    std::filesystem::path m_path;

    SearchPoint(SearchPointType type, uint8_t maxRecursiveDepth, std::filesystem::path path);

    bool isValid();

    void addDiscoveredFilesToSet(std::set<std::filesystem::path>& setToAddPathsTo);

};



/**
 * @file Shader_FileCatalog.h
 * @class FileCatalog
 * 
 * @brief The FileCatalog Class Is A Core Utility Within The Shader::ResourceManager Sub-Module Of
 * The Shader Library, Designed To Handle Resource Fetching By Managing And Searching Through A
 * Catalog Of File Paths. 
 * 
 * The Class Allows Users To Define Search Points (Directories Or Individual File Paths) And
 * Perform Recursive Or Non-Recursive Searches Based On Configurations. It Provides Several
 * Methods To Retrieve File Paths By Name, Extension, Or Both, As Well As Retrieve All Matches
 * For A Given Search Criteria. The FileCatalog Maintains A Centralized List Of Valid File Paths
 * That Is Continuously Updated As New Search Points Are Added, Ensuring Efficient And Reliable
 * Resource Management For Shader-Related Operations.
 */
class FileCatalog {

//Methods
public:
    /**
     * Constructor That Creates A FileCatalog Object With
     * A Root Directory Starting At The Current Working
     * Directory. By Default This Handles Recursive
     * Searching Up To A Depth Of 10, But Can Be Set
     * To 0 To Only Search From The Starting Directory
     * And Avoid Searching Any Sub-Directories
     * 
     * @param maxRecursiveDepth Maximum Recursive Depth To Search For Files And Add To File Catalog.
     * Example: 0 Signifys Only Searching Stating Directory, While 1 Would Search Starting Directory
     * Only It's Immediate Sub-Directories.
     */
    FileCatalog(uint8_t maxRecursiveDepth = 10);

    /**
     * Constructor That Creates A FileCatalog Object With
     * The Passed In Directory As The Starting Root
     * Directory. By Default This Handles Recursive
     * Searching Up To A Depth Of 10, But Can Be Set
     * To 0 To Only Search From The Starting Directory
     * And Avoid Searching Any Sub-Directories
     * 
     * @param rootDirectory File Path That Represents The Directory To Start The Search From 
     * When Creating The FileCatalog Object
     * 
     * @param maxRecursiveDepth Maximum Recursive Depth To Search For Files And Add To File Catalog.
     * Example: 0 Signifys Only Searching Stating Directory, While 1 Would Search Starting Directory
     * Only It's Immediate Sub-Directories.
     */
    FileCatalog(const std::filesystem::path& rootDirectory, uint8_t maxRecursiveDepth = 10);

    /**
     * Destructor That Handles Cleanup Of FileCatalog Object
     * This Destructor Just Explicitly Defines The Default
     * Destructor And Adds A Log Message To Record When
     * A FileCatalog Object Is Destroyed.
     */
    ~FileCatalog();
    
    /**
     * Searches Through The Available File Paths And Returns The First File Path
     * That Matches The Given Name. The Search Is Case Sensitive By Default.
     * 
     * @param name The Name Of The File To Search For.
     * @param caseSensitive Determines Whether The Search Should Be Case Sensitive.
     * Default Value Is `true`.
     * 
     * @return The First File Path Matching The Given Name. If No Match Is Found,
     * An Empty `std::filesystem::path` Object Is Returned.
     */
    std::filesystem::path getPathByName(const std::string& name, bool caseSensitive = true);

    /**
     * Searches Through The Available File Paths And Returns The First File Path
     * That Matches The Given File Extension. The Search Is Case Sensitive By Default.
     * 
     * @param extension The File Extension To Search For, Including The Leading Period (e.g., ".txt").
     * @param caseSensitive Determines Whether The Search Should Be Case Sensitive.
     * Default Value Is `true`.
     * 
     * @return The First File Path Matching The Given File Extension. If No Match Is Found,
     * An Empty `std::filesystem::path` Object Is Returned.
     */
    std::filesystem::path getPathByExtension(const std::string& extension, bool caseSensitive = true);

    /**
     * Searches Through The Available File Paths And Returns The First File Path
     * That Matches Both The Given Name And File Extension. The Search Is Case
     * Sensitive By Default.
     * 
     * @param name The Name Of The File To Search For.
     * @param extension The File Extension To Search For, Including The Leading Period (e.g., ".txt").
     * @param caseSensitive Determines Whether The Search Should Be Case Sensitive.
     * Default Value Is `true`.
     * 
     * @return The First File Path Matching Both The Given Name And File Extension.
     * If No Match Is Found, An Empty `std::filesystem::path` Object Is Returned.
     */
    std::filesystem::path getPathByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive = true);

    /**
     * Searches Through The Available File Paths And Returns All File Paths
     * That Match The Given Name. The Search Is Case Sensitive By Default.
     * 
     * @param name The Name Of The File To Search For.
     * @param caseSensitive Determines Whether The Search Should Be Case Sensitive.
     * Default Value Is `true`.
     * 
     * @return A `std::vector` Containing All File Paths That Match The Given Name.
     * If No Matches Are Found, An Empty Vector Is Returned.
     */
    std::vector<std::filesystem::path> getAllPathsByName(const std::string& name, bool caseSensitive = true);

    /**
     * Searches Through The Available File Paths And Returns All File Paths
     * That Match The Given File Extension. The Search Is Case Sensitive By Default.
     * 
     * @param extension The File Extension To Search For, Including The Leading Period (e.g., ".txt").
     * @param caseSensitive Determines Whether The Search Should Be Case Sensitive.
     * Default Value Is `true`.
     * 
     * @return A `std::vector` Containing All File Paths That Match The Given File Extension.
     * If No Matches Are Found, An Empty Vector Is Returned.
     */
    std::vector<std::filesystem::path> getAllPathsByExtension(const std::string& extension, bool caseSensitive = true);

    /**
     * Searches Through The Available File Paths And Returns All File Paths
     * That Match Both The Given Name And File Extension. The Search Is Case
     * Sensitive By Default.
     * 
     * @param name The Name Of The File To Search For.
     * @param extension The File Extension To Search For, Including The Leading Period (e.g., ".txt").
     * @param caseSensitive Determines Whether The Search Should Be Case Sensitive.
     * Default Value Is `true`.
     * 
     * @return A `std::vector` Containing All File Paths That Match Both The Given
     * Name And File Extension. If No Matches Are Found, An Empty Vector Is Returned.
     */
    std::vector<std::filesystem::path> getAllPathsByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive = true);

    /**
     * This Method Returns All Available Paths From Catalog.
     * 
     * @return A `std::vector` Containing All File Paths That Are Currently Available.
     */
    std::vector<std::filesystem::path> getAllAvailablePaths();

    /**
     * This Method Allows Adds The Following Directory As A Search
     * Spot To The File Catalog And Calls The `updateCatalog()`
     * Method To Search Through All SearchPoints To Update
     * Lists Of File Paths Available. This Method Assumes That
     * The Search For Files Is Non-Recursive From The SearchPoint
     * 
     * @param directory New Directory To Be Added As A SearchPoint
     * 
     * @return 
     *  - `True` If Directory Passed In Is Valid And Can Be Used As A SearchPoint.
     * 
     *  - `False` If Unable To Find Or Access New SearchPoint
     * 
     * @note New SearchPoint Is Still Added Even If Method Returns False, In The Event That The Directory Is
     * Created Or Accessible Later In The Program When `updateCatalog()` Is Called
     */
    bool addDirectoryAsSearchPoint(const std::filesystem::path& directory);

    /**
     * This Method Allows Adds The Following Directory As A Search
     * Spot To The File Catalog And Calls The `updateCatalog()`
     * Method To Search Through All SearchPoints To Update
     * Lists Of File Paths Available. This Method Uses The Passed
     * In maxRecursiveDepth Value To Determine How Deep Of A Recursive
     * Search Will Be Performed. By Default This Is Set To 10
     * 
     * @param directory New Directory To Be Added As A SearchPoint
     * 
     * @param maxRecursiveDepth Describes How Deep Of A Recursive Search Will Be Performed On New SearchPoint
     * 
     * @return True, If Directory Passed In Is Valid And Can Be Used As A SearchPoint
     * 
     * @return False, If Unable To Find Or Access New SearchPoint 
     * 
     * @note New SearchPoint Is Still Added Even If Method Returns False, In The Event That The Directory Is
     * Created Or Accessible Later In The Program When `updateCatalog()` Is Called
     */ 
    bool addDirectoryAsRecursiveSearchPoint(const std::filesystem::path& directory, uint8_t maxRecursiveDepth = 10);

    /**
     * This Method Adds The Following Path As A SearchPoint
     * For A Single Specific File. This Method Will Call
     * `updateCatalog` After A Successful Addition To Update
     * Available File Paths From All SearchPoints.
     * 
     * @param path New Path To Be Added As A SearchPoint
     * 
     * @return True, If Path Passed In Is Valid And Can Be Used As A SearchPoint
     * 
     * @return False, If Unable To Find Or Access New SearchPoint 
     * 
     * @note New SearchPoint Is Still Added Even If Method Returns False, In The Event That The Path Is
     * Created Or Accessible Later In The Program When `updateCatalog()` Is Called
     */
    bool addPathAsSearchPoint(const std::filesystem::path& path);

    /**
     * This Method Updates The Available File Paths
     * By Searching Through All SearchPoints According Their
     * Specifications And Types. This Method Is Called Automatically
     * Whenever Any New SearchPoints Are Added, But Can Also
     * Be Called Manually In Order To Re-Check The Validity
     * Of Existing File Paths Or Check For New Additional Ones
     */
    void updateCatalog();

private:


//Data
public:


private:
//Need to work out logic for sets and stuff next (also fix any comments that still say vector instead of set)
    std::set<std::filesystem::path> m_availableFilePaths; //Stores All Valid File Paths
    std::vector<SearchPoint> m_searchPoints; //Tracks All Directories And Individual Paths Being Searched
};

#endif