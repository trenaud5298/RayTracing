#ifndef SHADER_RESOURCEMANAGER_H
#define SHADER_RESOURCEMANAGER_H

#include <vector>
#include <filesystem>
#include <iostream>

/**
 * @file Shader_ResourceManager.h
 * @brief Namespace For Managing And Fetching Resources Required By The Shader Library.
 *
 * ================================================================================  
 * Resource Management Overview  
 * ================================================================================  
 * The `Shader::ResourceManager` Namespace Handles The Discovery, Organization, And Retrieval  
 * Of Resource Files Needed For The Shader Library. It Provides A Centralized System For  
 * Locating Files Within The Working Directory And Its Subdirectories, Simplifying Resource Access.  
 *  
 * Key Responsibilities:  
 *  
 * ---Initialization And Cleanup---
 * 
 *    - `Init()` Automatically Discovers And Catalogs Available Resources At Startup.  
 * 
 *    - `Shutdown()` Cleans Up Any Dynamically Allocated Memory Used During Resource Management.  
 *  
 * -------Resource Retrieval-------
 * 
 *    - Fetch Resources By Name, Extension, Or A Combination Of Both.  
 * 
 *    - Retrieve Either The First Matching File Or A Complete List Of Matches For Flexible Access.  
 *  
 * ================================================================================  
 * Important Notes  
 * ================================================================================  
 * 
 *  - Resource Searches Are Performed Within The Working Directory And Nested Subdirectories.  
 * 
 *  - Retrieval Methods Support Optional Case Sensitivity, Providing Flexibility For Different Use Cases.  
 * 
 *  - For Bulk Operations, Use Methods Returning All Matches To Process Multiple Resources Simultaneously.  
 */
namespace Shader::ResourceManager {
    /**
     * @brief Initializes The Resource Manager By Cataloging Resources.
     *
     * This Method Searches The Working Directory And Its Subdirectories To Identify  
     * All Available Resources. It Must Be Called Before Any Other Methods In This Namespace.  
     *
     * @return True If Initialization Succeeds, False Otherwise.
     */
    bool InitResourceManager();

    /**
     * @brief Shuts Down The Resource Manager And Cleans Up Memory.
     *
     * This Method Ensures All Dynamically Allocated Memory Is Released Properly.  
     * It Should Be Called During Library Shutdown.
     *
     * @return True If Shutdown Succeeds, False Otherwise.
     */
    bool ShutdownResourceManager();

    /**
     * @brief Refreshes All Available Resources By Searching Through All
     * Specified Search Directories
     * 
     * This Method Updates The FileCatalog With All Avaiable And Valid File
     * Paths That Are Found From All SearchPoints. This Method Is Called
     * Internally Everytime A New SearchPoint Is Added Through `addSearchDirectory()`
     * Or `addSearchPath()` And Is Only Available If Manual Updating Of
     * Resources Is Required Due To File Changes Mid-Program
     */
    void updateResources();

    /**
     * @brief Adds A New Search Directory As A SearchPoint To Resource Manager
     * 
     * This Method Adds The directoryPath As A New Search Point To Look
     * For Avaialable Resources With The Given maxRecursiveDepth Signifying
     * How Deep Of A Recursive Search Should Be Performed. A Depth Of
     * 0 Signifys To Not Search Recursively
     * 
     * @param directoryPath Path To Be Used As A New SearchPoint For Available Resources
     * 
     * @param maxRecursiveDetph Value Signifying How Deep Of A Recursive Search To Be Performed. 
     * 0 Represents No Recursive Searching
     * 
     * @return 
     *  -`True` If SearchPoint Is Considered Valid
     * 
     *  -`False` Otherwise
     * 
     * @note SearchPoint Is Still Added Even If Not Valid And Will Be Checked Everytime Avaialble
     * Resources Are Updated In The Event Of A Change That Results In The SearchPoint Being Valid Later On
     */
    bool addSearchDirectory(std::filesystem::path directoryPath, uint8_t maxRecursiveDepth = 10);


    /**
     * @brief Adds A New Path As A SearchPoint To Resource Manager
     * 
     * This Method Adds The Path As A New Search Point To Look
     * For An Avaialable Resources At The Specified Path.
     * 
     * @param Path Path To Be Used As A New SearchPoint For An Available Resource
     * 
     * @return 
     *  -`True` If SearchPoint Is Considered Valid
     * 
     *  -`False` Otherwise
     * 
     * @note SearchPoint Is Still Added Even If Not Valid And Will Be Checked Everytime Avaialble
     * Resources Are Updated In The Event Of A Change That Results In The SearchPoint Being Valid Later On
     */
    bool addSearchPath(std::filesystem::path path);

    /**
     * @brief Retrieves The First Resource That Matches The Given Name.
     *
     * Searches For A Resource With The Specified Name Within The Catalog.  
     * Supports Optional Case Sensitivity Controlled By The `caseSensitive` Parameter.
     *
     * @param name The Name Of The Resource To Search For.
     * @param caseSensitive Whether The Search Should Be Case Sensitive. Defaults To `true`.
     * @return The File Path Of The First Matching Resource, Or An Empty Path If None Found.
     */
    std::filesystem::path getResourceByName(const std::string& name, bool caseSensitive = true);

    /**
     * @brief Retrieves The First Resource That Matches The Given Extension.
     *
     * Searches For A Resource With The Specified File Extension Within The Catalog.  
     * Supports Optional Case Sensitivity For Extensions.
     *
     * @param extension The File Extension To Search For (e.g., ".ini", ".png"). Note: The `.` Is Expected To Be A Part Of The Passed In Extension String
     * @param caseSensitive Whether The Search Should Be Case Sensitive. Defaults To `true`.
     * @return The File Path Of The First Matching Resource, Or An Empty Path If None Found.
     */
    std::filesystem::path getResourceByExtension(const std::string& extension, bool caseSensitive = true);

    /**
     * @brief Retrieves The First Resource That Matches The Given Name And Extension.
     *
     * Searches For A Resource With The Specified Name And File Extension.  
     * Supports Optional Case Sensitivity For Both Name And Extension.
     *
     * @param name The Name Of The Resource To Search For.
     * @param extension The File Extension To Search For. Note: The `.` Is Expected To Be A Part Of The Passed In Extension String
     * @param caseSensitive Whether The Search Should Be Case Sensitive. Defaults To `true`.
     * @return The File Path Of The First Matching Resource, Or An Empty Path If None Found.
     */
    std::filesystem::path getResourceByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive = true);

    /**
     * @brief Retrieves All Resources That Match The Given Name.
     *
     * Returns A List Of All Resources In The Catalog That Match The Specified Name.  
     * Supports Optional Case Sensitivity For Names.
     *
     * @param name The Name Of The Resources To Search For.
     * @param caseSensitive Whether The Search Should Be Case Sensitive. Defaults To `true`.
     * @return A Vector Of File Paths For All Matching Resources.
     */
    std::vector<std::filesystem::path> getAllResourcesByName(const std::string& name, bool caseSensitive = true);

    /**
     * @brief Retrieves All Resources That Match The Given Extension.
     *
     * Returns A List Of All Resources In The Catalog That Match The Specified File Extension.  
     * Supports Optional Case Sensitivity For Extensions.
     *
     * @param extension The File Extension To Search For (e.g., ".ini", ".png"). Note: The `.` Is Expected To Be A Part Of The Passed In Extension String
     * @param caseSensitive Whether The Search Should Be Case Sensitive. Defaults To `true`.
     * @return A Vector Of File Paths For All Matching Resources.
     */
    std::vector<std::filesystem::path> getAllResourcesByExtension(const std::string& extension, bool caseSensitive = true);

    /**
     * @brief Retrieves All Resources That Match The Given Name And Extension.
     *
     * Returns A List Of All Resources In The Catalog That Match The Specified Name And File Extension.  
     * Supports Optional Case Sensitivity For Both Name And Extension.
     *
     * @param name The Name Of The Resources To Search For.
     * @param extension The File Extension To Search For. Note: The `.` Is Expected To Be A Part Of The Passed In Extension String
     * @param caseSensitive Whether The Search Should Be Case Sensitive. Defaults To `true`.
     * @return A Vector Of File Paths For All Matching Resources.
     */
    std::vector<std::filesystem::path> getAllResourcesByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive = true);

    /**
     * @brief Retrieves All Available Resources That Have Been Found
     * 
     * Returns A List Of All Resources In The Catalog.
     * 
     * @return A Vector Of File Paths For All Available Resources
     */
    std::vector<std::filesystem::path> getAllAvailableResources();
}

#endif