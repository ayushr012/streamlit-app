import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

schemes_data = {
    'Agri Udaan': {
    'website': 'http://www.agricoop.nic.in/',
    'description': 'Agri Udaan aims to provide support to farmers, Farmer Producer Organizations (FPOs), and Self Help Groups (SHGs) involved in transporting agricultural produce via air cargo. Eligible participants can receive up to 50% subsidy on air freight charges for domestic cargo and 30% for international cargo. Additionally, they gain access to cold chain facilities and packaging infrastructure at airports, along with assistance with market linkages and branding.',
    'eligibility': 'Open to all farmers, Farmer Producer Organizations (FPOs), and Self Help Groups (SHGs) involved in transporting agricultural produce via air cargo.',
    'benefits': 'Up to 50% subsidy on air freight charges for domestic cargo and 30% for international cargo. Access to cold chain facilities and packaging infrastructure at airports. Assistance with market linkages and branding.',
    'limitations': 'Limited availability of air cargo infrastructure, especially in smaller towns. Stringent quality and packaging requirements for certain commodities. Not all agricultural produce is eligible for the subsidy.',
    'when to consider': "Throughout the year. However, specific deadlines might exist for certain air cargo terminals or commodities. It's best to check with the nearest airport or the official website for updates."
},
'Pradhan Mantri Krishi Sinchai Yojana (PMKSY)': {
    'website': 'https://pmksy.nic.in/',
    'description': 'Pradhan Mantri Krishi Sinchai Yojana (PMKSY) offers financial assistance, grants, and subsidies for constructing irrigation canals, drip irrigation systems, sprinkler systems, water harvesting structures, and farm ponds. It aims to improve irrigation infrastructure, enhance water availability and coverage, and reduce water usage through technologies like drip and sprinkler irrigation. Increased crop yields and productivity are expected outcomes of improved water management practices under PMKSY.',
    'eligibility': 'Individual farmers, farmer groups, and panchayats can apply for different components of PMKSY. Specific program components might have different eligibility criteria based on landholding size, crop type, and location.',
    'benefits': 'Financial assistance: Grants and subsidies for constructing irrigation canals, drip irrigation systems, sprinkler systems, water harvesting structures, and farm ponds. Improved irrigation infrastructure: Access to reliable and efficient irrigation systems for enhanced water availability and coverage. Reduced water usage: Water-saving technologies like drip and sprinkler irrigation minimize water wastage and optimize water use. Increased crop yields and productivity: Improved water availability and efficient irrigation lead to better plant growth and higher crop yields.',
    'limitations': 'Program accessibility and benefits might vary across states and regions. Farmers need to contribute a share of the project cost for some components. Proper maintenance and efficient water management practices are crucial for long-term sustainability.',
    'when to consider': 'Farmers facing challenges with irrigation water availability, insufficient access to irrigation infrastructure, or inefficient water use practices. Farmers in areas with inadequate irrigation systems or prone to droughts. Farmers willing to adopt water-saving irrigation technologies and improve water management.'
},

    'Soil Health Card Scheme': {
        'website': 'https://soilhealth.dac.gov.in/',
        'description': 'This scheme aims to provide farmers with detailed information about the health of their soil to improve agricultural productivity and sustainability.',
        'eligibility': 'All farmers cultivating any land are eligible. No specific documents or applications required.',
        'benefits': 'Free soil testing with detailed reports on nutrient levels and recommendations for balanced fertilization. Improved fertilizer use and enhanced soil health.',
        'when to consider': 'All farmers, regardless of size or specialization, who want to understand their soil health status and make informed decisions about fertilizer and nutrient management.'
    },
    'Krishi Vigyan Kendra (KVK)': {
    'website': 'https://kvk.icar.gov.in/',
    'description': 'This scheme provides agricultural extension services including training, workshops, demonstrations, and expert advice to farmers.',
    'eligibility': 'Open to all farmers, farmer groups, and individuals seeking knowledge and skill improvement in agriculture.',
    'benefits': 'Free training: Workshops, demonstrations, and field visits on diverse agricultural topics. Expert advice: Consult with scientists and specialists on crop selection, pest management, etc. Technology access: Learn about and potentially test new technologies like precision agriculture tools. Market linkages: KVKs may connect farmers with buyers and market information.',
    'limitations': 'Services may vary depending on the specific KVK and its resources. Proactive participation is needed to benefit from training programs and advice. May not be readily available in all remote areas.',
    'when to consider': 'Always: KVKs offer continuous learning and support, beneficial for farmers at any stage. Seeking knowledge and skill improvement: Enhance farming practices, adopt new technologies, or diversify operations. Facing specific challenges: Need guidance on pest control, soil management, or efficient water usage.'
    },
    'Hortinet Farmer Connect App': {
    'website': 'https://hortharyana.gov.in/en',
    'description': 'This app connects farmers directly with buyers, provides real-time market information, facilitates online ordering of agricultural inputs, and offers market intelligence and advisory services.',
    'eligibility': 'Open to all farmers and agriculture stakeholders.',
    'benefits': 'Connects farmers directly with buyers, eliminating middlemen. Provides access to real-time market information and prices. Facilitates online ordering and delivery of agricultural inputs. Offers market intelligence and advisory services.',
    'limitations': 'Requires smartphone and internet access, which may not be readily available in all rural areas. Reliance on technology may pose challenges for some farmers. Information accuracy and buyer reliability need to be monitored.'
    },
    'Mission for Integrated Development of Horticulture (MIDH)': {
    'website': 'https://nhb.gov.in/schemes.aspx',
    'description': 'This scheme aims to promote the integrated development of horticulture by providing financial assistance and support for infrastructure development, input purchase, technology adoption, capacity building, and marketing initiatives.',
    'eligibility': 'Individual farmers, joint liability groups, farmer producer organizations (FPOs), cooperatives. Land ownership or lease agreement for horticulture activities. Willingness to adopt improved technologies and marketing practices.',
    'benefits': 'Financial assistance for infrastructure development like nurseries, packing houses, cold storage. Support for input purchase, technology adoption, capacity building, and marketing initiatives. Grant-in-aid for production, post-harvest handling, and value addition projects.',
    'limitations': 'Scheme availability might vary depending on state and specific crop focus. Project proposals undergo competitive selection process. Utilization of funds and adherence to implementation guidelines are crucial.',
    'when to consider': 'Farmers looking to diversify into horticulture (fruits, vegetables, flowers, spices). Individuals or groups aiming to establish, modernize, or expand horticulture units. Those interested in adopting scientific cultivation practices and value addition techniques.'
    },
    'E-RaKAM': {
    'website': 'https://www.medianama.com/2021/08/223-eshram-national-database-unorganized-workers/',
    'description': 'The e-RaKAM platform allows farmers with valid PAN cards and bank accounts to connect with buyers across the country through online auctions. It offers improved price discovery, reduced dependence on middlemen, and a transparent and secure bidding process.',
    'eligibility': 'All farmers with valid PAN cards and bank accounts can register on the e-RaKAM platform.',
    'benefits': 'Direct access to buyers: Connect with buyers across the country through online auctions. Improved price discovery: Get competitive prices for your produce based on real-time bids. Reduced dependence on middlemen: Eliminate market intermediaries and gain greater control over selling price. Transparent and secure bidding process: Verified buyers and secure online platform ensure fair transactions.',
    'limitations': 'Requires digital literacy and access to internet and smartphones. May not be suitable for perishable or small-scale farm produce due to logistics and minimum quantity requirements. Farmers need to invest in proper grading and packaging infrastructure to meet platform standards.',
    'when to consider': 'Marketable surplus produce: Have sufficient quantity of good quality produce to sell beyond local markets. Digital access: Comfortable using smartphones and the internet for online bidding and platform navigation. Willing to meet quality standards: Grade and package produce according to specified requirements.'
    },
    'Integrated Scheme for Agriculture Marketing (ISAM)': {
    'website': 'https://www.indiafilings.com/learn/integrated-scheme-for-agricultural-marketing/',
    'description': 'The Integrated Scheme for Agriculture Marketing (ISAM) provides financial assistance and support for infrastructure development, market research, branding initiatives, promotional activities, and logistics development. It aims to improve agricultural market infrastructure and marketing efficiency.',
    'eligibility': 'Registered FPOs, cooperatives, marketing bodies, public-private partnerships. Focus on specific agricultural commodities and market development plans. Willingness to adopt transparent and efficient marketing practices.',
    'benefits': 'Financial assistance for infrastructure development like grading, sorting, packaging units. Support for market research, branding initiatives, promotional activities, and logistics development. Grant-in-aid for capacity building, market intelligence systems, and digital marketing platforms.',
    'limitations': 'Scheme availability and specific focus might differ depending on state and agricultural commodity. Project proposals undergo competitive selection process. Effective market linkages, brand development, and operational sustainability are essential.',
    'when to consider': 'Farmer producer organizations (FPOs), marketing cooperatives, agri-businesses. Individuals or groups aiming to improve agricultural market infrastructure and marketing efficiency. Those seeking to promote direct marketing, branding, and value addition initiatives.'
    },
    'Zero Hunger Mission': {
    'website': 'http://agricoop.nic.in/',
    'description': 'The Zero Hunger Mission targets marginalized farmers, landless laborers, and vulnerable communities by providing financial assistance for agricultural activities, infrastructure development, and capacity building. It promotes food security, nutritional well-being, and sustainable agricultural practices.',
    'eligibility': 'Varies depending on the sub-scheme, but generally targets marginalized farmers, landless laborers, and vulnerable communities.',
    'benefits': 'Provides financial assistance for agricultural activities, infrastructure development, and capacity building. Promotes food security and nutritional well-being. Encourages sustainable agricultural practices.',
    'limitations': 'Complex bureaucratic procedures can hinder access for some beneficiaries. Uneven distribution of benefits across states and regions. Sustainability of the program depends on continued government funding.'
},
'Unified Package Insurance Scheme (UPIS)': {
    'website': 'https://pmfby.gov.in/',
    'description': 'The Unified Package Insurance Scheme (UPIS) offers comprehensive coverage against loss or damage to crops due to various natural calamities. It provides low premium rates with government subsidy and an easy claim settlement process.',
    'eligibility': 'All farmers owning land in Gurugram are eligible, irrespective of the size of their holding.',
    'benefits': 'Comprehensive coverage against loss or damage to crops due to various natural calamities, including weather events, pests, and diseases. Low premium rates with government subsidy. Easy claim settlement process.',
    'limitations': 'Limited coverage for certain crops and losses. Farmers need to provide accurate land records and crop details for enrollment. Delays in claim settlement might occur in some cases.',
    'when to consider': 'Anytime during the Kharif or Rabi season. The specific window for Gurugram might vary, so check with the nearest agriculture department office for confirmation.'
},
'Gramin Agriculture Markets (GRAMs)': {
    'website': 'Varies by state.',
    'description': 'Gramin Agriculture Markets (GRAMs) provide stall allocation, infrastructure support, marketing and branding assistance, and direct interaction with consumers. They prioritize small and marginal farmers, SC/ST communities, and women.',
    'eligibility': 'Individual farmers or Farmer Producer Organizations (FPOs) registered with the state agriculture department. Prioritization for small and marginal farmers, SC/ST communities, and women.',
    'benefits': 'Stall allocation in GRAMs markets: Access to dedicated space for showcasing and selling produce. Infrastructure support: Cold storage facilities, processing units, and packaging materials may be available. Marketing and branding assistance: Promotion of farmer products within the GRAMs network. Direct interaction with consumers: Build customer relationships and receive feedback.',
    'limitations': 'Availability of GRAMs markets may vary depending on location. Requires commitment to regular market participation and product quality maintenance. May not be suitable for small-scale or perishable produce due to market logistics.',
    'when to consider': 'Direct selling opportunities: Farmers with marketable surplus produce seeking direct connection with consumers. Value-added products: Offer processed or packaged agricultural products (e.g., spices, oils, handicrafts) for higher returns. Local market access: Located in areas with existing GRAMs infrastructure and consumer base.'
},
'E-NAM': {
    'website': 'https://enam.gov.in/',
    'description': 'E-NAM provides wider market access, transparent pricing, reduced transaction costs, and real-time market information to farmers. All farmers with PAN cards and mobile numbers can register, subject to specific requirements for listed commodities and market participation.',
    'eligibility': 'All farmers with PAN cards and mobile numbers can register on the e-NAM platform. Specific requirements for listed commodities and market participation might apply, such as minimum quantity requirements for bidding.',
    'benefits': 'Wider market access: Connect with buyers across the country, bypassing local middlemen and potentially fetching higher prices. Transparent pricing: Online auction system ensures competitive bidding and fair market pricing. Reduced transaction costs: Direct connection with buyers eliminates intermediary charges and minimizes transportation costs. Real-time market information: Access to commodity prices and market trends on the platform.',
    'limitations': 'Need for internet access and basic digital literacy: Farmers need computer/smartphone and basic online skills to navigate the platform. Limited commodity coverage: Not all agricultural products are listed on e-NAM, and requirements might vary between states. Logistics and payment challenges: Farmers need to arrange for transportation and ensure secure payment after online sale.',
    'when to consider': 'Farmers with excess produce beyond local market needs looking for wider marketing options and potentially better prices. Farmers comfortable with basic online platforms and willing to engage in direct online bidding and transactions. Producers of agricultural commodities listed on the e-NAM platform in their state (varies by state).'
},
'Project Chaman': {
    'website': 'Varies by state.',
    'description': 'Project Chaman offers financial assistance, training, support in sourcing bees, and marketing assistance to individual farmers or farmer groups involved in beekeeping activities.',
    'eligibility': 'Individual farmers or farmer groups involved in beekeeping activities.',
    'benefits': 'Financial assistance: Up to 75% subsidy on beehives, equipment, and training. Training: Learn beekeeping methods, honey production techniques, and hive management. Support in sourcing bees: Access to quality bee colonies at subsidized rates. Marketing assistance: Connect with honey buyers and participate in honey fairs.',
    'limitations': 'Available primarily in areas with suitable bee forage and climatic conditions. Beekeeping requires dedicated time and effort for hive maintenance and honey extraction. Initial investment needed for purchasing beehives and equipment.',
    'when to consider': 'Interested in beekeeping: Diversify income, utilize land for honey production, or contribute to pollination. Access to suitable resources: Availability of flowering plants or dedicated bee forage area. Willing to invest time and effort: Beekeeping requires regular hive maintenance and honey extraction.'
},
'Scheme for Management of Crop Residues (SMCR)': {
    'website': 'https://www.teriin.org/sites/default/files/2020-01/crop-residue-management.pdf',
    'description': 'SMCR provides financial assistance, machinery support, and grants for setting up residue collection and utilization centers to individuals, farmer groups, cooperatives, and entrepreneurs. Eligible participants must have land ownership or lease agreements for crop residue management activities and be willing to adopt residue management techniques like mulching, composting, or utilization.',
    'eligibility': 'Individual farmers, farmer groups, cooperatives, entrepreneurs. Land ownership or lease agreement for crop residue management activities. Willingness to adopt residue management techniques like mulching, composting, or utilization.',
    'benefits': 'Financial assistance for purchase of machinery like shredders, balers, composting units. Support for setting up residue collection and utilization centers. Grant-in-aid for awareness campaigns, capacity building, and value addition projects.',
    'limitations': 'Scheme availability and specific focus might differ depending on state and crop residue type. Project proposals undergo competitive selection process. Sustainable residue management and adherence to environmental regulations are crucial.',
    'when to consider': 'Farmers facing challenges in managing crop residues like burning. Those interested in adopting in-situ or ex-situ residue management practices. Individuals or groups looking to utilize crop residues for value addition activities.'
},
'Paramparagat Krishi Vikas Yojana (PKVY)': {
    'website': 'https://darpg.gov.in/sites/default/files/Paramparagat%20Krishi%20Vikas%20Yojana.pdf',
    'description': 'PKVY offers financial assistance, technical support, and market linkage to individual farmers, Farmer Producer Organizations (FPOs), and Self Help Groups (SHGs) interested in transitioning to organic farming. Eligible applicants must meet specific landholding size requirements and commit to adopting organic farming practices.',
    'eligibility': 'Individual farmers, Farmer Producer Organizations (FPOs), and Self Help Groups (SHGs) can apply. Landholding size might have specific requirements (e.g., 2 ha in clusters for individual farmers).',
    'benefits': 'Financial assistance: Grants for organic inputs, certification fees, and capacity building. Technical support: Training on organic farming practices, composting, and pest management. Market linkage: Support for connecting with organic buyers and marketing networks.',
    'limitations': 'Organic farming requires knowledge, commitment, and initial adaptation period. Availability of organic inputs and markets might be limited in some areas. Transition to organic certification can take 2-3 years.',
    'when to consider': 'Farmers interested in transitioning to organic farming, aiming to improve soil health and fertility, and potentially fetching premium prices for organic produce. Farmers with suitable land and resources for organic cultivation, like access to organic inputs and markets.'
},
'Centralized Farm Machinery Performance Portal (CFMPP)': {
    'website': 'https://dbt.mpdage.org/',
    'description': 'CFMPP offers farmers in India access to information on various farm machinery available for rent or purchase. It allows users to compare performance specifications and rental rates from different service providers and facilitates direct contact with machinery owners or rental companies. The portal is open to all farmers in India.',
    'eligibility': 'Open to all farmers in India.',
    'benefits': 'Provides information on different types of farm machinery available for rent or purchase. Allows comparison of performance specifications and rental rates from different service providers. Facilitates direct contact with machinery owners or rental companies.',
    'limitations': 'Availability of machinery might vary depending on location and demand. Farmers need to have internet access and basic computer skills to navigate the portal effectively.'
},
'Sub Mission on Agricultural Mechanization (SMAM)': {
    'website': 'http://www.agricoop.nic.in/',
    'description': 'SMAM provides financial subsidy, training, and capacity building support to individual farmers, farmer groups, and Custom Hiring Centres (CHCs) for adopting agricultural machinery. Eligibility criteria may include landholding size and specific machinery requirements. The mission aims to modernize farm operations, promote CHCs, and enhance productivity and efficiency in agriculture.',
    'eligibility': 'Individual farmers, farmer groups, and Custom Hiring Centres (CHCs) can apply. Landholding size and type of machinery might have specific eligibility criteria.',
    'benefits': 'Financial subsidy: Up to 40% of the cost of machinery and equipment. Training and capacity building: Skill development programs on operation and maintenance of machinery. Promotion of CHCs: Encouragement for establishing CHCs to provide machinery services to small and marginal farmers.',
    'limitations': 'Availability of specific machinery might vary depending on location and program allocation. Farmers need to contribute a portion of the machinery cost. Proper maintenance and technical knowledge are crucial for effective utilization.',
    'when to consider': 'Farmers looking to modernize their farm operations by adopting machinery for land preparation, sowing, planting, harvesting, threshing, or processing. Farmers facing labor shortage or aiming to improve efficiency and productivity. Farmers cultivating crops suitable for mechanization with sufficient landholding size.'
},
'Mission Fingerling': {
    'website': 'Varies by state.',
    'description': 'Mission Fingerling aims to support registered fish farmers with subsidies, training, fingerling supply, and marketing assistance. Priority is given to small and marginal farmers, Scheduled Castes/Tribes, and women. The mission focuses primarily on freshwater fish species like carp and catfish.',
    'eligibility': 'Registered fish farmers with the state fisheries department. Priority given to small and marginal farmers, Scheduled Castes/Tribes, and women.',
    'benefits': 'Subsidies: Up to 50% subsidy on hatchery construction, equipment, and fingerling purchase. Training: Learn fish breeding and production techniques from fisheries department experts. Fingerling supply: Access to quality fingerlings at subsidized rates for stocking ponds. Marketing assistance: Connect with fish markets and processing units to sell produce.',
    'limitations': 'Subsidy amounts and availability may vary depending on state and budget allocation. Strict adherence to pond management guidelines and biosecurity protocols is required. Focuses primarily on freshwater fish species like carp and catfish.',
    'when to consider': 'Existing fish farmers: Expand operations by establishing hatcheries or increasing fingerling production. New fish farmers: Interested in starting fish farming with government support. Areas with suitable water resources: Availability of ponds, wells, or canals for fish farming.'
},
'National Scheme for Sustainable Agriculture (NSSA)': {
    'website': 'http://www.agricoop.nic.in/',
    'description': 'NSSA offers financial assistance, technical support, and training programs to all farmers interested in adopting sustainable agricultural practices. It aims to improve soil health, reduce environmental impact, and enhance farm resilience to climate change and weather extremes.',
    'eligibility': 'All farmers, regardless of size or specialization, can benefit from NSSA initiatives and programs. Specific programs within NSSA might have individual eligibility criteria based on crop, location, and project type.',
    'benefits': 'Financial assistance: Grants and subsidies for adopting sustainable practices, purchasing equipment, and participating in training programs. Technical support: Training and capacity building on sustainable agriculture techniques, soil health management, and efficient resource utilization. Improved soil health and fertility: Sustainable practices lead to better soil quality, enhanced nutrient retention, and increased crop yields. Reduced environmental impact: Lower water usage, minimized chemical use, and improved soil organic matter benefit the environment. Enhanced farm resilience: Sustainable practices make farms more resilient to climate change and weather extremes.',
    'limitations': 'Program availability and specific benefits might vary across states and seasons. Farmers need to adopt recommended practices effectively to reap the full benefits. Transition to sustainable practices might require initial adaptation and potentially lower yields in the short term.',
    'when to consider': 'Farmers interested in adopting sustainable agricultural practices like integrated pest management, organic farming, agroforestry, water conservation, and precision agriculture. Farmers aiming to improve soil health, reduce environmental impact, and enhance long-term farm productivity. Farmers in areas where NSSA interventions and programs are being implemented.'
},
'Sahakar Pragya': {
    'website': 'https://www.dhyeyaias.in/current-affairs/daily-current-affairs/ministry-of-co-operation',
    'description': 'Sahakar Pragya provides financial assistance, capacity building, and grant-in-aid to registered Farmer Producer Organizations (FPOs) aiming to strengthen their operations, infrastructure, and market linkages. Eligible FPOs must have a minimum of 100 farmer members, a track record of successful agricultural activities, and a willingness to adopt professional governance and business practices.',
    'eligibility': 'Registered FPOs with minimum 100 farmer members. Track record of successful agricultural activities and financial stability. Willingness to adopt professional governance and business practices.',
    'benefits': 'Financial assistance for setting up infrastructure like grading, sorting, packaging units. Support for capacity building, branding, market research, and value chain development. Grant-in-aid for operational expenses, professional fees, and market development initiatives.',
    'limitations': 'FPOs need to demonstrate strong governance, financial viability, and market potential. Project proposals undergo stringent evaluation and approval process. Effective utilization of funds and adherence to scheme guidelines are essential.',
    'when to consider': 'Farmer producer organizations (FPOs) aiming to strengthen their operations and business management. FPOs looking to build processing and value addition infrastructure. Those seeking to improve market linkages and access for member farmers.'
},
'AGMARKNET Portal': {
    'website': 'Varies by state.',
    'description': 'AGMARKNET Portal empowers farmers by providing increased participation in shaping agricultural policies, enhanced access to resources, knowledge sharing, and promotion of inclusive development. It allows farmers to access real-time commodity prices, arrival information, and price trends across major wholesale markets, aiding in price research, decision-making, and exploring international markets.',
    'eligibility': 'None specified.',
    'benefits': 'Empowerment and voice: Increased participation in shaping agricultural policies and programs relevant to their needs. Enhanced access to resources: Potential for improved access to infrastructure, funding, and technical support based on identified needs. Knowledge sharing and learning: Exchange best practices and traditional knowledge with other farmers and experts. Promote inclusive development: Contribution to a more equitable and sustainable agricultural future for the community.',
    'limitations': 'Requires sustained commitment and active participation in group discussions and planning activities. Success depends on effective collaboration and leadership within the community. Outcomes may not be immediate, and tangible benefits may take time to materialize.',
    'when to consider': 'Price and market trend research: Access real-time commodity prices, arrival information, and price trends across major wholesale markets. Decision-making support: Utilize price data to plan harvest and sale periods for optimal returns. Exploring international markets: Access information on agricultural exports and imports for informed commodity selection.'
},
'Pradhan Mantri Fasal Bima Yojana (PMFBY)': {
    'website': 'https://pmfby.gov.in/',
    'description': 'Pradhan Mantri Fasal Bima Yojana (PMFBY) provides financial compensation to farmers cultivating notified crops in notified areas for crop losses exceeding a pre-determined threshold due to covered risks. Both loanee and non-loanee farmers are eligible, with loanee farmers automatically enrolled. While it offers reduced risk and income stability by providing financial protection against crop failures, farmers need to pay premiums based on crop, area, and coverage options. Some exclusions apply, and claim settlement processes might vary depending on the insurance provider and loss assessment.',
    'eligibility': 'All farmers cultivating notified crops in the notified areas can apply. Loanee and non-loanee farmers are eligible (loanee farmers have automatic enrollment).',
    'benefits': 'Financial compensation: Payment for crop losses exceeding a pre-determined threshold due to covered risks. Reduced risk and income stability: Provides financial protection against crop failures and ensures income continuity.',
    'limitations': 'Farmers need to pay premiums based on crop, area, and coverage options. Some exclusions apply (e.g., negligence, pre-existing damage). Claim settlement processes might vary depending on the insurance provider and loss assessment.',
    'when to consider': 'Farmers facing risks from natural calamities (floods, droughts, hailstorms), pest and disease outbreaks, or significant price fluctuations. Farmers growing notified crops (varies by state) within the stipulated sowing period.'
},
'Accelerated Pulses Production Programme (APPP)': {
    'website': 'http://www.agricoop.nic.in/',
    'description': 'Accelerated Pulses Production Programme (APPP) provides financial assistance, technical support, and market linkage to individual farmers and farmer groups cultivating pulses. Eligible farmers receive grants and subsidies for seeds, fertilizers, bio-pesticides, and other inputs. Training on improved cultivation practices and post-harvest management is also provided. While the program aims to increase pulse production and productivity, farmers need to adhere to recommended practices and effectively utilize inputs.',
    'eligibility': 'Individual farmers and farmer groups cultivating pulses can apply. Specific program components might have different eligibility criteria based on crop and location.',
    'benefits': 'Financial assistance: Grants and subsidies for seeds, fertilizers, bio-pesticides, and other inputs. Technical support: Training on improved cultivation practices, pest and disease management, and post-harvest management. Market linkage: Support for accessing markets and fair prices for pulse produce.',
    'limitations': 'Program availability and benefits might vary across states and seasons. Farmers need to adhere to recommended practices and utilize inputs effectively. Market access support might depend on local market dynamics and infrastructure.',
    'when to consider': 'Farmers specializing in pulse crops (chickpeas, lentils, etc.) looking to increase production and productivity. Farmers in areas with suitable agro-climatic conditions for pulse cultivation. Farmers willing to adopt improved seeds, technologies, and integrated pest management practices.'
},
'PM-AASHA': {
    'description': 'PM-AASHA provides financial assistance and infrastructure support to farmer groups, cooperatives, and individual farmers for post-harvest management, value addition, and market access improvement. Eligible entities can apply through state governments or project implementing agencies. While the program offers grants for up to 85% of the project cost, with farmers contributing the remaining share, it aims to enhance post-harvest management, value addition, and marketability of produce through improved infrastructure and processing units.',
    'eligibility': 'Farmer groups, cooperatives, and individual farmers can apply through state governments or project implementing agencies. Specific project types might have eligibility criteria based on landholding size, group composition, and project feasibility.',
    'benefits': 'Financial assistance: Grants for up to 85% of the project cost, with farmers contributing the remaining share. Improved post-harvest management: Reduces losses, enhances product quality, and potentially fetches higher prices. Value addition: Processing units enable farmers to add value to their produce and increase income. Enhanced market access: Improved infrastructure facilitates better storage, transportation, and marketing of produce.',
    'when to consider': 'Farmer groups, cooperatives, or individual farmers with specific infrastructure needs for post-harvest management, value addition, and improved marketability. Projects for rural agri-infrastructure development like storage facilities, processing units, farm machinery, and market yards.'
},
'Meghdoot App': {
    'website': 'https://apps.mgov.gov.in/details;jsessionid=488905E85792F83B97C40C13106244F8?appid=1517',
    'description': 'The Meghdoot App primarily targets farmers in rain-fed areas and those affected by natural disasters, providing real-time weather updates, forecasts, and crop advisories. It helps farmers make informed decisions about sowing, irrigation, and pest control based on weather conditions and soil moisture levels. However, the accuracy of weather data and advisories may vary depending on location, and the app requires smartphone and internet access, similar to Hortinet. It may have limited reach in areas with poor mobile network coverage.',
    'eligibility': 'Primarily for farmers in rain-fed areas and those affected by natural disasters.',
    'benefits': 'Provides real-time weather updates and forecasts. Offers crop advisories based on weather conditions and soil moisture. Helps farmers make informed decisions about sowing, irrigation, and pest control.',
    'limitations': 'Accuracy of weather data and advisories may vary depending on location. Requires smartphone and internet access, similar to Hortinet. Limited reach in areas with poor mobile network coverage.'
}
}

def suggest_similar_schemes(input_text, tfidf_matrix, input_tfidf, n=3):
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    similar_scheme_indices = similarities.argsort()[0][-n:][::-1]
    similar_scheme_names = [list(schemes_data.keys())[i] for i in similar_scheme_indices]
    return similar_scheme_names

def get_similar_schemes(input_text, n=3):
    tfidf_matrix, input_tfidf = extract_info(input_text)
    similar_schemes = suggest_similar_schemes(input_text, tfidf_matrix, input_tfidf, n)
    return similar_schemes

def extract_info(input_text):
    scheme_texts = []
    for data in schemes_data.values():
        text = ' '.join(data.values())
        scheme_texts.append(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(scheme_texts)
    input_tfidf = vectorizer.transform([input_text])
    return tfidf_matrix, input_tfidf

def main():
    st.title("Scheme Recommendation System")
    user_input = st.text_input("Enter your query:")
    
    if st.button("Get Similar Schemes"):
        if user_input:
            similar_schemes = get_similar_schemes(user_input)
            st.subheader("Similar Schemes:")
            for scheme in similar_schemes:
                st.write("- ", scheme)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()







