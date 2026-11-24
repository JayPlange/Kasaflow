\# KasaFlow engineering evidence



\*\*Snapshot date:\*\* 24 November 2026  

\*\*Repository:\*\* https://github.com/JayPlange/Kasaflow  

\*\*Status:\*\* Active development; pre-launch



This document records the current engineering evidence and its limits. It is not a claim of production deployment, commercial traction, or independently verified customer outcomes.



\## Product and integration scope



KasaFlow is a WhatsApp commerce workflow system being developed ahead of a pilot with Adom De Jeweller, a jewellery retailer in Ghana.



The current repository documents:



\- Python and FastAPI application.

\- WhatsApp Cloud API integration.

\- WooCommerce catalogue and order integration.

\- Product discovery, recommendation, pricing, delivery and quotation tools.

\- English and Twi voice pathways, plus image-based product matching workflows.

\- Application-level authentication, rate limiting, webhook signature verification and logging.

\- An order proposal flow separated from order confirmation.



The current README describes eleven registered tools. The language model selects tools; application code owns pricing, validation, state and order execution.



\## Order authorisation design



The key boundary is between proposing an order and creating one.



\- `propose\_order()` prepares and prices an order proposal.

\- `confirm\_order()` is the order-writing path.

\- The application revalidates the quoted variant and total before committing.

\- WooCommerce does not provide idempotency-key support, so the application uses an order metadata token to check whether a timed-out write already succeeded before retrying.

\- Customer order lookup and cancellation are scoped to the WhatsApp session's phone identity.

\- Prompt-injection guardrails and human escalation are implemented in application code.



These are documented implementation decisions. They should not be interpreted as proof that every possible failure mode has been eliminated.



\## Test and evaluation evidence



The current repository README reports \*\*937 passing automated tests\*\*.



The separate behavioural evaluator runs multi-turn conversations through the customer-routing pipeline and checks expected tools, fields and responses across fifteen categories. The evaluator documentation currently records \*\*two operator-test conversations\*\* in its seed replay set. It explicitly says there are no real customer conversations in that set.



The repository also contains an architecture audit dated 20 August 2026 and a conversation-evaluation workflow with recorded and live replay modes. The audit's findings and fixes should be read with their dates in mind; some items have since changed. For example, the current README says the per-session turn lock has been implemented.



\## Current stage and limits



\- The project is in active development and testing ahead of a jewellery retailer pilot.

\- The pilot has not yet launched.

\- The application is not deployed to a public host.

\- KasaFlow has not generated revenue.

\- The two-conversation seed replay is not a representative customer dataset.

\- Test counts establish checks performed in the repository, not production reliability or commercial performance.



\## Source documents



\- \[Repository README](../README.md)

\- \[Architecture audit, 20 August 2026](architecture\_audit\_2026-08-20.md)

\- \[Conversation evaluation workflow](../evalset/README.md)



For current implementation details and reproducible commands, use the source files and scripts in this repository. Preserve raw evaluation reports separately; do not replace them with manually reconstructed summaries.

