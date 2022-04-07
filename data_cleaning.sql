/*
Cleaning housing data in MySQL

Data used can be found here:
https://github.com/AlexTheAnalyst/PortfolioProjects/blob/main/Nashville%20Housing%20Data%20for%20Data%20Cleaning.xlsx
*/

SELECT * FROM projects.nashville_housing_data;

--------------------------------------------------------------------------------------------------------------------------
-- Standardize Date Format

-- Convert SaleDate column from text into a date type
ALTER TABLE projects.nashville_housing_data
	ADD COLUMN SaleDateConverted Date AFTER SaleDate;
UPDATE projects.nashville_housing_data SET
	SaleDateConverted = STR_TO_DATE(SaleDate,'%M %e, %Y');

--------------------------------------------------------------------------------------------------------------------------
-- Populate Property Address data

-- Note there are a number of rows missing address information
Select *
From projects.nashville_housing_data
Where PropertyAddress  = ''
order by ParcelID;

-- Some Properties have been sold multiple times, each sale creates a new row in our table with a unique ID.
-- However, when a property is sold it is very unusual for its address to change. By matching
-- property IDs and differentiating UniqueIDs we can fill in the missing address information
Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress
From projects.nashville_housing_data a
JOIN projects.nashville_housing_data b
	on a.ParcelID = b.ParcelID
	AND a.UniqueID  <> b.UniqueID 
Where a.PropertyAddress = ''
ORDER by a.ParcelID;

-- fill the missing property address values
UPDATE projects.nashville_housing_data AS a
INNER JOIN (SELECT PropertyAddress, ParcelID,UniqueID FROM projects.nashville_housing_data) AS b 
	ON a.ParcelID = b.ParcelID
	AND a.UniqueID <> b.UniqueID
SET a.PropertyAddress = b.PropertyAddress
WHERE a.PropertyAddress = '';

-- Quick check
Select *
From projects.nashville_housing_data
Where PropertyAddress  = ''
order by ParcelID;

--------------------------------------------------------------------------------------------------------------------------
-- Breaking out Address into Individual Columns (Address, City, State)

ALTER TABLE projects.nashville_housing_data
Add PropertySplitAddress char(255),
Add PropertySplitCity char(255);

Update projects.nashville_housing_data 
SET PropertySplitAddress = SUBSTRING_INDEX(PropertyAddress,',',1 ),
	PropertySplitCity = SUBSTRING_INDEX(PropertyAddress,',',-1 );

-- Breaking out OwnerAddress
-- Note that owner address includes the state of residence, otherwise same structure as property address
Select OwnerAddress
From projects.nashville_housing_data;

ALTER TABLE projects.nashville_housing_data
Add OwnerSplitAddress char(255),
Add OwnerSplitCity char(255),
Add OwnerSplitState char(255);

Update projects.nashville_housing_data
SET OwnerSplitAddress = SUBSTRING_INDEX(OwnerAddress,',',1 ),
	OwnerSplitCity = SUBSTRING_INDEX(SUBSTRING_INDEX(OwnerAddress,',',-2 ),',',1 ),
	OwnerSplitState = SUBSTRING_INDEX(OwnerAddress,',',-1 );

--------------------------------------------------------------------------------------------------------------------------

-- Change Y and N to Yes and No in "Sold as Vacant" field

-- notice that SoldAs Vacant has four responses, we'll convert them to 2 
Select Distinct(SoldAsVacant), Count(SoldAsVacant)
From projects.nashville_housing_data
Group by SoldAsVacant
order by 2;

Update projects.nashville_housing_data
SET SoldAsVacant = 
CASE When SoldAsVacant = 'Y' THEN 'Yes'
	 When SoldAsVacant = 'N' THEN 'No'
	 ELSE SoldAsVacant
	 END;
 
-----------------------------------------------------------------------------------------------------------------------------------------------------------

-- Remove Duplicates

WITH RowNumCTE AS(
Select *, ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
                 OwnerName,
				 LegalReference
				 ORDER BY
					UniqueID
					) row_num
From projects.nashville_housing_data)

/* For examining the CTE
Select *
From RowNumCTE
Where row_num > 1;
*/

Delete from projects.nashville_housing_data
Using projects.nashville_housing_data Join RowNumCTE ON projects.nashville_housing_data.UniqueID = RowNumCTE.UniqueID
WHERE RowNumCTE.row_num > 1;

