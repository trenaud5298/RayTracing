#include <Shader/Shader_FileCatalog.h>
#include <Shader/Shader_Logging.h>
#include <algorithm>
#include <cctype>
#include <Shader/Shader_Timer.h>





#pragma region Static Methods

static void searchDirectoryAndAddFilesToSet(std::set<std::filesystem::path>& setToAddPathsTo, const std::filesystem::path& directoryToSearch, uint8_t maxRecursiveDepth) {
    if( !std::filesystem::exists(directoryToSearch) || !std::filesystem::is_directory(directoryToSearch) ) {
        Shader::Logging::log(LOG_TYPE_DEBUG | LOG_TYPE_ERROR, "Invalid Directory Was Passed Into searchDirectoryAndAddFilesToSet()! Check isValid() Method Which Should Ensure This Before Being Called");
        return;
    }

    std::filesystem::recursive_directory_iterator recursiveIterator(directoryToSearch) ;
    std::filesystem::recursive_directory_iterator end = std::filesystem::recursive_directory_iterator();

    //Handle Recursive Search While Also Limiting Depth Of Search To Max Recursive Depth
    while(recursiveIterator != end) {
        //If Current Depth Is Greater Than Max Depth, Do Not Continue Traversal, And Move Onto Next Iteration
        if(recursiveIterator.depth() > maxRecursiveDepth) {
            recursiveIterator.pop(); //Skips Any Further Traversal
            continue;
        }

        if(recursiveIterator->is_regular_file()) {
            setToAddPathsTo.insert(recursiveIterator->path());
        }

        ++recursiveIterator;
    }



}


static bool isEqual(const std::string& lowerCaseString, const std::string& comparedString) {
    if(lowerCaseString.size() != comparedString.size()) {
        return false;
    }

    return std::equal(lowerCaseString.begin(), lowerCaseString.end(), comparedString.begin(),
        [](unsigned char lowerChar, unsigned char compareChar) {
            return lowerChar == std::tolower(compareChar);
        }
    );
    
}

static std::string processStringToLowerCase(const std::string& string) {
    std::string processedString = string;
    std::transform(processedString.begin(), processedString.end(), processedString.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return processedString;
}
#pragma endregion
















#pragma region SearchPoint


SearchPoint::SearchPoint(SearchPointType type, uint8_t maxRecursiveDepth, std::filesystem::path path) :
m_type(type), m_maxRecursiveDepth(maxRecursiveDepth), m_path(path) {
    //Creates A New Search Point
}

bool SearchPoint::isValid() {
    //Each Block Here Represents A Potential Check That May Need To Be Ran To Check If The SearchPoint Is Valid
    //Exceptions Catch Reasons For Path Being Invalid Such As Invalid Permissions Or Other General Issues
    try {
        if(!std::filesystem::exists(m_path)) {
            std::string errorMessage = "SearchPoint Path Does Not Exists: " + m_path.string();
            Shader::Logging::log(LOG_TYPE_ERROR, errorMessage);
            return false;
        }

        if(m_type == SearchPointType::DIRECTORY) {
            if(!std::filesystem::is_directory(m_path)) {
                std::string errorMessage = "SearchPoint Exists But Does Not Match Expected Type(Directory): " + m_path.string();
                Shader::Logging::log(LOG_TYPE_ERROR, errorMessage);
                return false;
            }
        }

        if(m_type == SearchPointType::INDIVIDUAL_PATH) {
            if(!std::filesystem::is_regular_file(m_path)) {
                std::string errorMessage = "SearchPoint Exists But Does Not Match Expected Type(Individual Path): " + m_path.string();
                Shader::Logging::log(LOG_TYPE_ERROR, errorMessage);
                return false;
            }
        }

    } catch (const std::filesystem::filesystem_error& e) {
        std::string errorMessage = "Filesystem Error: " + std::string(e.what());
        Shader::Logging::log(LOG_TYPE_ERROR, errorMessage);
        return false;

    } catch (const std::exception& e) {
        std::string errorMessage = "General Error: " + std::string(e.what());
        Shader::Logging::log(LOG_TYPE_ERROR, errorMessage);
        return false;
    }

    return true;
}

void SearchPoint::addDiscoveredFilesToSet(std::set<std::filesystem::path>& setToAddPathsTo) {
    std::vector<std::filesystem::path> filePaths;
    if(!isValid()) {
        Shader::Logging::log(LOG_TYPE_WARNING, "SearchPoint At Path " + m_path.string() + " Is Invalid! Cannot Pull File Paths! See Log Above For More Details!");
        return;
    }

    switch(m_type) {
        case DIRECTORY:
            searchDirectoryAndAddFilesToSet(setToAddPathsTo, m_path, m_maxRecursiveDepth);
            break;

        case INDIVIDUAL_PATH:
            setToAddPathsTo.insert(m_path);
            break;

        default:
            Shader::Logging::log(LOG_TYPE_DEBUG | LOG_TYPE_ERROR, "Impossible Case Reached In SearchPoint::getAllFilePaths(), Default Case For Switch Statements Should Be Impossible To Reach");
            break;
    }

}


#pragma endregion








#pragma region File Catalog
FileCatalog::FileCatalog(uint8_t maxRecursiveDepth) : FileCatalog(std::filesystem::current_path(), maxRecursiveDepth){
    //Calls Next Constructor With The Working Directory As The Root Directory
}

FileCatalog::FileCatalog(const std::filesystem::path& rootDirectory, uint8_t maxRecursiveDepth) {
    addDirectoryAsRecursiveSearchPoint(rootDirectory, maxRecursiveDepth);
    Shader::Logging::log(LOG_TYPE_INFO, "FileCatalog Created");
}


FileCatalog::~FileCatalog() {
    Shader::Logging::log(LOG_TYPE_INFO, "FileCatalog Destroyed");
}

std::filesystem::path FileCatalog::getPathByName(const std::string& name, bool caseSensitive) {
    std::filesystem::path result;
    if(caseSensitive) {
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(name == possibleResult.stem().string()) {
                result = possibleResult;
                break;
            }
        }    
    } else {
        std::string searchTerm = processStringToLowerCase(name);
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(isEqual(searchTerm, possibleResult.stem().string())) {
                result = possibleResult;
                break;
            }
        }
    }
    return result;
}

std::filesystem::path FileCatalog::getPathByExtension(const std::string& extension, bool caseSensitive) {
    std::filesystem::path result;
    if(caseSensitive) {
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(extension == possibleResult.extension().string()) {
                result = possibleResult;
                break;
            }
        }    
    } else {
        std::string searchTerm = processStringToLowerCase(extension);
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(isEqual(searchTerm, possibleResult.extension().string())) {
                result = possibleResult;
                break;
            }
        }
    }
    return result;
}

std::filesystem::path FileCatalog::getPathByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive) {
    std::filesystem::path result;
    std::string searchTerm = name + extension;
    if(caseSensitive) {
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(searchTerm == possibleResult.filename().string()) {
                result = possibleResult;
                break;
            }
        }    
    } else {
        searchTerm = processStringToLowerCase(searchTerm);
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(isEqual(searchTerm, possibleResult.filename().string())) {
                result = possibleResult;
                break;
            }
        }
    }
    return result;
}

std::vector<std::filesystem::path> FileCatalog::getAllPathsByName(const std::string& name, bool caseSensitive) {
    std::vector<std::filesystem::path> results;
    if(caseSensitive) {
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(name == possibleResult.stem().string()) {
                results.push_back(possibleResult);
            }
        }    
    } else {
        std::string searchTerm = processStringToLowerCase(name);
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(isEqual(searchTerm, possibleResult.stem().string())) {
                results.push_back(possibleResult);
            }
        }
    }
    return results;
}

std::vector<std::filesystem::path> FileCatalog::getAllPathsByExtension(const std::string& extension, bool caseSensitive) {
    Timer<std::chrono::microseconds> timer("Search Timer");
    std::vector<std::filesystem::path> results;
    if(caseSensitive) {
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(extension == possibleResult.extension().string()) {
                results.push_back(possibleResult);
            }
        }    
    } else {
        std::string searchTerm = processStringToLowerCase(extension);
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(isEqual(searchTerm, possibleResult.extension().string())) {
                results.push_back(possibleResult);
            }
        }
    }
    return results;
}

std::vector<std::filesystem::path> FileCatalog::getAllPathsByNameAndExtension(const std::string& name, const std::string& extension, bool caseSensitive) {
    std::vector<std::filesystem::path> results;
    std::string searchTerm = name + extension;
    if(caseSensitive) {
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(searchTerm == possibleResult.filename().string()) {
                results.push_back(possibleResult);
            }
        }    
    } else {
        searchTerm = processStringToLowerCase(searchTerm);
        for(const std::filesystem::path& possibleResult : m_availableFilePaths) {
            if(isEqual(searchTerm, possibleResult.filename().string())) {
                results.push_back(possibleResult);
            }
        }
    }
    return results;
}

std::vector<std::filesystem::path> FileCatalog::getAllAvailablePaths() {
    return std::vector<std::filesystem::path>(m_availableFilePaths.begin(), m_availableFilePaths.end());
}

bool FileCatalog::addDirectoryAsSearchPoint(const std::filesystem::path& directory) {
    return addDirectoryAsRecursiveSearchPoint(directory, 0); //Calls Next Method With A Max Recursive Depth Of 0
}

bool FileCatalog::addDirectoryAsRecursiveSearchPoint(const std::filesystem::path& directory, uint8_t maxRecursiveDepth) {
    SearchPoint newSearchPoint = SearchPoint(SearchPointType::DIRECTORY, maxRecursiveDepth, directory);
    m_searchPoints.push_back(newSearchPoint);
    updateCatalog();
    return newSearchPoint.isValid();
}

bool FileCatalog::addPathAsSearchPoint(const std::filesystem::path& path) {
    SearchPoint newSearchPoint = SearchPoint(SearchPointType::INDIVIDUAL_PATH, 0, path);
    m_searchPoints.push_back(newSearchPoint);
    updateCatalog();
    return newSearchPoint.isValid();
}

void FileCatalog::updateCatalog() {
    Timer<std::chrono::milliseconds> timer("update timer");
    m_availableFilePaths.clear();
    Shader::Logging::log(LOG_TYPE_INFO, "Updating Catalog");
    for(SearchPoint searchPoint : m_searchPoints) {
        Shader::Logging::log(LOG_TYPE_INFO, "Searching SearchPoint With Path: " + searchPoint.m_path.string());
        searchPoint.addDiscoveredFilesToSet(m_availableFilePaths);
    }
}
#pragma endregion