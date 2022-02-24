/*
Covid 19 Data Exploration in mysql
Skills used: Joins, CTE's, Temp Tables, Windows Functions, Aggregate Functions, Creating Views, Converting Data Types

Data from: https://ourworldindata.org/covid-deaths
*/

-- turn off safe updates to edit tables
SET SQL_SAFE_UPDATES=0;

/*=================================================================================================
Convert the date column from "mm/dd/yy" to "yyyy-mm-dd" DATE type for the "coviddeaths" table
=================================================================================================*/

-- add columns to house the date substrings
ALTER TABLE projects.coviddeaths
ADD COLUMN month VARCHAR(2) AFTER date,
ADD COLUMN day VARCHAR(2) AFTER date,
ADD COLUMN year VARCHAR(4) AFTER date;

/* Breakdown date into 3 columns for year, month, and day. Goal is to create a new date column in the
   yyyy-mm-dd format so that we can convert it into DATE format */
UPDATE projects.coviddeaths SET 
month = SUBSTRING_INDEX(date,'/',1), 
day = SUBSTRING_INDEX(SUBSTRING_INDEX(date,'/',2),'/',-1),
year =SUBSTRING_INDEX(date,'/',-1);

-- add 20 to year column, and 0 to day & month if = 1-9
UPDATE projects.coviddeaths SET
year = CONCAT('20',year);

UPDATE projects.coviddeaths SET 
month = CONCAT('0',month) 
WHERE LENGTH(month)<2;

UPDATE projects.coviddeaths SET 
day = CONCAT('0',day) 
WHERE LENGTH(day)<2;

-- Drop the old date column, and concat the year, month, day columns back together
UPDATE projects.coviddeaths SET
date = CONCAT_WS('-',year,month,day);

/* Convert the date column to type DATETIME and then to DATE. Unsure why, but when I tried to convert from varchar 
   to date it did not work. However varchar -> datetime -> date, worked fine. This will reduce the size of the date 
   column from 8 bytes to 3 */
UPDATE projects.coviddeaths SET
date = CONVERT(date, DATETIME),
date = CONVERT(date, DATE);

/*================================================================================================
Convert the date column from "mm/dd/yy" to "yyyy-mm-dd" DATE type for the "covidvaccination" table
================================================================================================*/

ALTER TABLE projects.covidvaccinations
ADD COLUMN month VARCHAR(2) AFTER date,
ADD COLUMN day VARCHAR(2) AFTER date,
ADD COLUMN year VARCHAR(4) AFTER date;

UPDATE projects.covidvaccinations SET 
month = SUBSTRING_INDEX(date,'/',1), 
day = SUBSTRING_INDEX(SUBSTRING_INDEX(date,'/',2),'/',-1),
year =SUBSTRING_INDEX(date,'/',-1);

UPDATE projects.covidvaccinations SET
year = CONCAT('20',year);

UPDATE projects.covidvaccinations SET 
month = CONCAT('0',month) 
WHERE LENGTH(month)<2;

UPDATE projects.covidvaccinations SET 
day = CONCAT('0',day) 
WHERE LENGTH(day)<2;

UPDATE projects.covidvaccinations SET
date = CONCAT_WS('-',year,month,day);

UPDATE projects.covidvaccinations SET
date = CONVERT(date, DATETIME),
date = CONVERT(date, DATE);

/*==============================================================================================
  Begin data exploration
===============================================================================================*/ 
SELECT * 
FROM projects.coviddeaths
ORDER BY 3,4 ;

SELECT location, date, total_cases, new_cases, total_deaths, population
FROM projects.coviddeaths
ORDER BY 1,2;

-- looking at total cases vs total deaths in the United states
SELECT location, date, total_cases, total_deaths, (total_deaths/total_cases)* 100 as MortalityRate
FROM projects.coviddeaths
WHERE location like '%states%'
ORDER BY 1,2;

-- looking at total cases vs population in the United states
SELECT location, date, total_cases, population, (total_cases/population)* 100 as PercentPopulationInfected
FROM projects.coviddeaths
WHERE location like '%states%'
ORDER BY 1,2;

-- looking at countries with highest infection rate compared to population
SELECT location, population, MAX(total_cases) AS HighestInfectionCount, MAX((total_cases/population))*100 as
	PercentPopulationInfected
FROM projects.coviddeaths
GROUP BY location, population
ORDER BY PercentPopulationInfected DESC;

-- Countries with highest death count
SELECT location, MAX(total_deaths) as TotalDeathCount
FROM projects.coviddeaths
WHERE continent is not null
GROUP BY location
ORDER BY TotalDeathCount DESC;

-- Continents with highest death count
SELECT continent, MAX(total_deaths) as TotalDeathCount
FROM projects.coviddeaths
WHERE continent is not null
GROUP BY continent
ORDER BY TotalDeathCount desc;

-- global numbers
SELECT date, SUM(new_cases) as total_cases, SUM(new_deaths) as total_deaths, SUM(new_deaths)/SUM(new_cases)*100 as DeathPercentage
FROM projects.coviddeaths
WHERE continent is not null
GROUP BY date
ORDER BY 1,2;

-- Looking at total population vs vaccination
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.date) 
	AS rolling_vaccination_count
FROM projects.coviddeaths d
JOIN projects.covidvaccinations v
	on d.location = v.location
    and d.date = v.date
WHERE d.continent is not null
order by 2,3;

-- Using CTE to perform Calculation on Partition By in previous query
With PopsvsVac (continent, location, date, population, new_vaccinations, rolling_vaccination_count) as 
(
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.date) AS rolling_vaccination_count
FROM projects.coviddeaths d
JOIN projects.covidvaccinations v
	on d.location = v.location
    and d.date = v.date
WHERE d.continent is not null
)
SELECT * ,(rolling_vaccination_count/population)*100
FROM PopsvsVac;

-- Using Temp Table to perform Calculation on Partition By in previous query
DROP Table if exists PercentPopulationVaccinated;
CREATE TEMPORARY Table PercentPopulationVaccinated
(
Continent varchar(255),
Location varchar(255),
Date datetime,
Population bigint,
New_vaccinations bigint,
rolling_vaccination_count bigint
);
INSERT INTO PercentPopulationVaccinated
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.date) AS rolling_vaccination_count
FROM projects.coviddeaths d
JOIN projects.covidvaccinations v
	on d.location = v.location
    and d.date = v.date
WHERE d.continent is not null;

Select *, (rolling_vaccination_count/population)*100
From PercentPopulationVaccinated;

-- Creating View to store data for later visualizations
Create View PercentPopulationVaccinated as
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.date) AS rolling_vaccination_count
FROM projects.coviddeaths d
JOIN projects.covidvaccinations v
	on d.location = v.location
    and d.date = v.date
WHERE d.continent is not null;