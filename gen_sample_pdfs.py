from __future__ import annotations

import textwrap
from pathlib import Path

import fitz

OUT = Path(__file__).resolve().parent / "data" / "pdfs"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = {
    "Renewable_Energy_Handbook": {
        "intro": "This handbook documents principles and operations of renewable energy systems.",
        "topics": {
            "Solar Photovoltaics": "A photovoltaic cell converts sunlight directly into electricity via the photovoltaic effect. Crystalline silicon panels typically achieve 18 to 22 percent efficiency. The peak sun hours metric determines daily energy yield.",
            "Wind Turbines": "A horizontal-axis wind turbine extracts kinetic energy from moving air. The Betz limit caps theoretical efficiency at 59.3 percent. Cut-in wind speed is typically 3 to 4 meters per second.",
            "Battery Storage": "Lithium-iron-phosphate batteries offer long cycle life and thermal stability. Depth of discharge affects longevity; keeping it below 80 percent extends cycle count.",
            "Grid Integration": "Inverters synchronize distributed generation with grid frequency, nominally 50 or 60 hertz. Curtailment occurs when supply exceeds demand.",
        },
    },
    "Clinical_Nutrition_Guide": {
        "intro": "A reference on human nutrition, macronutrients, and dietary guidelines.",
        "topics": {
            "Macronutrients": "Carbohydrates provide 4 kilocalories per gram, proteins 4, and fats 9. The acceptable macronutrient distribution range for carbohydrates is 45 to 65 percent of energy.",
            "Vitamins": "Vitamin D regulates calcium absorption. Deficiency causes rickets in children and osteomalacia in adults. The recommended daily allowance is 600 international units for adults.",
            "Hydration": "Adequate water intake supports thermoregulation and nutrient transport. General guidance suggests about 2 to 3 liters daily depending on activity.",
            "Glycemic Index": "The glycemic index ranks carbohydrate foods by their effect on blood glucose. Low-GI foods are below 55 and promote stable energy.",
        },
    },
    "Network_Security_Manual": {
        "intro": "This manual covers network security fundamentals and defensive practices.",
        "topics": {
            "Firewalls": "A stateful firewall tracks connection state and filters packets accordingly. Default-deny policies block all traffic except explicitly allowed flows.",
            "Encryption": "TLS 1.3 secures data in transit using ephemeral key exchange for forward secrecy. AES-256 is a widely used symmetric cipher.",
            "Authentication": "Multi-factor authentication combines something you know, have, and are. It substantially reduces account takeover risk.",
            "Intrusion Detection": "An intrusion detection system inspects traffic for known signatures or anomalous behavior and raises alerts for analysts.",
        },
    },
    "Macroeconomics_Primer": {
        "intro": "An introduction to macroeconomic concepts and policy tools.",
        "topics": {
            "GDP": "Gross domestic product measures the total value of goods and services produced within a country in a period. Real GDP adjusts for inflation.",
            "Inflation": "Inflation is a sustained rise in the general price level. Central banks often target around 2 percent annual inflation.",
            "Monetary Policy": "Central banks adjust policy interest rates to influence borrowing, spending, and inflation. Lower rates stimulate demand.",
            "Unemployment": "The unemployment rate is the share of the labor force actively seeking work. Structural unemployment stems from skill mismatches.",
        },
    },
    "Cell_Biology_Notes": {
        "intro": "Notes on cell structure, function, and molecular biology.",
        "topics": {
            "Mitochondria": "Mitochondria generate ATP through oxidative phosphorylation. They contain their own circular DNA inherited maternally.",
            "DNA Replication": "DNA polymerase synthesizes a new strand in the 5-prime to 3-prime direction. Replication is semiconservative.",
            "Cell Membrane": "The plasma membrane is a phospholipid bilayer that regulates transport via embedded proteins. It is selectively permeable.",
            "Protein Synthesis": "Ribosomes translate messenger RNA into polypeptides. Transfer RNA delivers amino acids matching each codon.",
        },
    },
    "Climate_Science_Report": {
        "intro": "A report summarizing climate science, drivers, and impacts.",
        "topics": {
            "Greenhouse Effect": "Greenhouse gases such as carbon dioxide and methane trap outgoing infrared radiation, warming the lower atmosphere.",
            "Carbon Cycle": "Carbon moves between the atmosphere, oceans, soil, and biosphere. Oceans absorb roughly a quarter of human carbon emissions.",
            "Sea Level Rise": "Thermal expansion and melting ice sheets raise sea levels. Observed rise has accelerated over recent decades.",
            "Mitigation": "Mitigation reduces emissions through renewable energy, efficiency, and reforestation. Adaptation manages unavoidable impacts.",
        },
    },
    "Software_Architecture_Text": {
        "intro": "A text on software architecture patterns and trade-offs.",
        "topics": {
            "Microservices": "Microservices decompose an application into independently deployable services communicating over the network. They increase operational complexity.",
            "Caching": "Caching stores frequently accessed data closer to consumers to cut latency. Cache invalidation is a notoriously hard problem.",
            "Load Balancing": "A load balancer distributes requests across servers to improve availability and throughput. Round-robin and least-connections are common strategies.",
            "Event Sourcing": "Event sourcing persists state changes as an immutable sequence of events, enabling reconstruction and audit.",
        },
    },
    "World_History_Survey": {
        "intro": "A survey of major periods and events in world history.",
        "topics": {
            "Industrial Revolution": "The Industrial Revolution began in Britain in the late 18th century, mechanizing production with steam power and factories.",
            "Roman Empire": "The Roman Empire reached its greatest territorial extent under Trajan in 117 AD, spanning three continents.",
            "Printing Press": "Gutenberg's movable-type printing press, around 1440, dramatically lowered the cost of books and spread literacy.",
            "Renaissance": "The Renaissance was a cultural movement from the 14th to 17th centuries emphasizing humanism, art, and classical learning.",
        },
    },
    "Pharmacology_Compendium": {
        "intro": "A compendium of pharmacological principles and drug classes.",
        "topics": {
            "Pharmacokinetics": "Pharmacokinetics describes absorption, distribution, metabolism, and excretion of drugs. Half-life determines dosing frequency.",
            "Analgesics": "Nonsteroidal anti-inflammatory drugs inhibit cyclooxygenase enzymes to reduce pain and inflammation.",
            "Antibiotics": "Beta-lactam antibiotics disrupt bacterial cell wall synthesis. Resistance arises through beta-lactamase enzymes.",
            "Dosage": "Therapeutic index is the ratio between toxic and effective dose. A narrow index requires careful monitoring.",
        },
    },
    "Astronomy_Lecture_Notes": {
        "intro": "Lecture notes covering stars, planets, and cosmology.",
        "topics": {
            "Stellar Fusion": "Stars fuse hydrogen into helium in their cores, releasing energy that counteracts gravitational collapse.",
            "Black Holes": "A black hole is a region where gravity is so strong that not even light escapes past the event horizon.",
            "Redshift": "Cosmological redshift stretches light from receding galaxies, evidence that the universe is expanding.",
            "Exoplanets": "Exoplanets are detected via transit dimming and radial-velocity wobble of their host stars.",
        },
    },
}

PAGES_PER_DOC = 210


def build_pdf(name: str, spec: dict):
    pdf = fitz.open()
    sections = list(spec["topics"].items())
    for i in range(PAGES_PER_DOC):
        page = pdf.new_page()
        title, text = sections[i % len(sections)]
        content = (
            f"{name.replace('_', ' ')}\n"
            f"Section: {title}    (page {i + 1})\n\n"
            f"{spec['intro']}\n\n"
            f"{text}\n\n"
            + "\n".join(
                textwrap.wrap(
                    f"Detailed discussion {i + 1}: {text} "
                    f"This page elaborates on {title.lower()} with worked context so "
                    f"that retrieval over a large corpus returns precise, citable passages. "
                    f"Key term: {title}. Reference value and explanation repeated for depth.",
                    width=95,
                )
            )
        )
        page.insert_text((54, 60), content, fontsize=11, fontname="helv")
    dest = OUT / f"{name}.pdf"
    pdf.save(str(dest))
    pdf.close()
    return dest


def main():
    print(f"Generating {len(DOMAINS)} PDFs ({PAGES_PER_DOC} pages each) into {OUT}")
    for name, spec in DOMAINS.items():
        result = build_pdf(name, spec)
        print(f"  ✓ {result.name}  ({PAGES_PER_DOC} pages)")
    print("Done. Now run the app and click 'Ingest data/pdfs/'.")


if __name__ == "__main__":
    main()
